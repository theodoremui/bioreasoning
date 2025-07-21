import requests
import time
import pandas as pd
import logging

from sqlalchemy import Engine, create_engine, Connection, Result
from typing import Optional, Dict, Any, List, Literal, Union, cast

# Set up logging
logger = logging.getLogger(__name__)


class OtelTracesSqlEngine:
    def __init__(
        self,
        engine: Optional[Engine] = None,
        engine_url: Optional[str] = None,
        table_name: Optional[str] = None,
        service_name: Optional[str] = None,
    ):
        self.service_name: str = service_name or "service"
        self.table_name: str = table_name or "otel_traces"
        self._connection: Optional[Connection] = None
        if engine:
            self._engine: Engine = engine
        elif engine_url:
            self._engine = create_engine(url=engine_url)
        else:
            raise ValueError("One of engine or engine_setup_kwargs must be set")

    def _connect(self) -> None:
        try:
            self._connection = self._engine.connect()
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def _export(
        self,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        url = "http://localhost:16686/api/traces"
        params: Dict[str, Union[str, int]] = {
            "service": self.service_name,
            "start": start_time
            or int(time.time() * 1000000) - (24 * 60 * 60 * 1000000),
            "end": end_time or int(time.time() * 1000000),
            "limit": limit or 1000,
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to export traces from Jaeger: {e}")
            # Return empty data structure to prevent crashes
            return {"data": []}
        except Exception as e:
            logger.error(f"Unexpected error during trace export: {e}")
            return {"data": []}

    def _to_pandas(self, data: Dict[str, Any]) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []

        # Loop over each trace
        for trace in data.get("data", []):
            trace_id = trace.get("traceID")
            service_map = {
                pid: proc.get("serviceName")
                for pid, proc in trace.get("processes", {}).items()
            }

            for span in trace.get("spans", []):
                span_id = span.get("spanID")
                operation = span.get("operationName")
                start = span.get("startTime")
                duration = span.get("duration")
                process_id = span.get("processID")
                service = service_map.get(process_id, "")
                status = next(
                    (
                        tag.get("value")
                        for tag in span.get("tags", [])
                        if tag.get("key") == "otel.status_code"
                    ),
                    "",
                )
                parent_span_id = None
                if span.get("references"):
                    parent_span_id = span["references"][0].get("spanID")

                rows.append(
                    {
                        "trace_id": trace_id,
                        "span_id": span_id,
                        "parent_span_id": parent_span_id,
                        "operation_name": operation,
                        "start_time": start,
                        "duration": duration,
                        "status_code": status,
                        "service_name": service,
                    }
                )

        return pd.DataFrame(rows)

    def _to_sql(
        self,
        dataframe: pd.DataFrame,
        if_exists_policy: Optional[Literal["fail", "replace", "append"]] = None,
    ) -> None:
        try:
            if not self._connection:
                self._connect()
            dataframe.to_sql(
                name=self.table_name,
                con=self._connection,
                if_exists=if_exists_policy or "append",
            )
        except Exception as e:
            logger.error(f"Failed to write traces to database: {e}")
            # Don't raise the exception to prevent application crashes

    def to_sql_database(
        self,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: Optional[int] = None,
        if_exists_policy: Optional[Literal["fail", "replace", "append"]] = None,
    ) -> None:
        try:
            data = self._export(start_time=start_time, end_time=end_time, limit=limit)
            df = self._to_pandas(data=data)
            if not df.empty:
                self._to_sql(dataframe=df, if_exists_policy=if_exists_policy)
            else:
                logger.info("No trace data to export")
        except Exception as e:
            logger.error(f"Failed to export traces to database: {e}")

    def execute(
        self,
        statement: Any,
        parameters: Optional[Any] = None,
        execution_options: Optional[Any] = None,
        return_pandas: bool = False,
    ) -> Union[Result, pd.DataFrame]:
        try:
            if not self._connection:
                self._connect()
            if not return_pandas:
                self._connection = cast(Connection, self._connection)
                return self._connection.execute(
                    statement=statement,
                    parameters=parameters,
                    execution_options=execution_options,
                )
            return pd.read_sql(sql=statement, con=self._connection)
        except Exception as e:
            logger.error(f"Failed to execute SQL statement: {e}")
            raise

    def to_pandas(
        self,
    ) -> pd.DataFrame:
        try:
            if not self._connection:
                self._connect()
            return pd.read_sql_table(table_name=self.table_name, con=self._connection)
        except Exception as e:
            logger.error(f"Failed to read traces from database: {e}")
            # Return empty DataFrame to prevent crashes
            return pd.DataFrame()

    def disconnect(self) -> None:
        try:
            if not self._connection:
                raise ValueError("Engine was never connected!")
            self._engine.dispose(close=True)
        except Exception as e:
            logger.error(f"Failed to disconnect from database: {e}")
            raise
