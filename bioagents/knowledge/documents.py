from dataclasses import dataclass
from sqlalchemy import (
    Table,
    MetaData,
    Column,
    Text,
    Integer,
    create_engine,
    Engine,
    Connection,
    insert,
    select,
)
from typing import Optional, List, cast, Union


def apply_string_correction(string: str) -> str:
    return string.replace("''", "'").replace('""', '"')


@dataclass
class ManagedDocument:
    document_name: str
    content: str
    summary: str
    q_and_a: str
    mindmap: str
    bullet_points: str


class DocumentManager:
    def __init__(
        self,
        engine: Optional[Engine] = None,
        engine_url: Optional[str] = None,
        table_name: Optional[str] = None,
        table_metadata: Optional[MetaData] = None,
    ):
        self.table_name: str = table_name or "documents"
        self._table: Optional[Table] = None
        self._connection: Optional[Connection] = None
        self.metadata: MetaData = cast(MetaData, table_metadata or MetaData())
        if engine or engine_url:
            self._engine: Union[Engine, str] = cast(
                Union[Engine, str], engine or engine_url
            )
        else:
            raise ValueError("One of engine or engine_setup_kwargs must be set")

    @property
    def connection(self) -> Connection:
        if not self._connection:
            self._connect()
        return cast(Connection, self._connection)

    @property
    def table(self) -> Table:
        if self._table is None:
            self._create_table()
        return cast(Table, self._table)

    def _connect(self) -> None:
        # move network calls outside of constructor
        if isinstance(self._engine, str):
            self._engine = create_engine(self._engine)
        self._connection = self._engine.connect()

    def _create_table(self) -> None:
        self._table = Table(
            self.table_name,
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("document_name", Text),
            Column("content", Text),
            Column("summary", Text),
            Column("q_and_a", Text),
            Column("mindmap", Text),
            Column("bullet_points", Text),
        )
        self._table.create(self.connection, checkfirst=True)

    def put_documents(self, documents: List[ManagedDocument]) -> None:
        for document in documents:
            stmt = insert(self.table).values(
                document_name=document.document_name,
                content=document.content,
                summary=document.summary,
                q_and_a=document.q_and_a,
                mindmap=document.mindmap,
                bullet_points=document.bullet_points,
            )
            self.connection.execute(stmt)
        self.connection.commit()

    def get_documents(self, names: Optional[List[str]] = None) -> List[ManagedDocument]:
        if self.table is None:
            self._create_table()
        if not names:
            stmt = select(self.table).order_by(self.table.c.id)
        else:
            stmt = (
                select(self.table)
                .where(self.table.c.document_name.in_(names))
                .order_by(self.table.c.id)
            )
        result = self.connection.execute(stmt)
        rows = result.fetchall()
        documents = []
        for row in rows:
            documents.append(
                ManagedDocument(
                    document_name=row.document_name,
                    content=row.content,
                    summary=row.summary,
                    q_and_a=row.q_and_a,
                    mindmap=row.mindmap,
                    bullet_points=row.bullet_points,
                )
            )
        return documents

    def get_names(self) -> List[str]:
        if self.table is None:
            self._create_table()
        stmt = select(self.table)
        result = self.connection.execute(stmt)
        rows = result.fetchall()
        return [row.document_name for row in rows]

    def disconnect(self) -> None:
        if not self._connection:
            raise ValueError("Engine was never connected!")
        if isinstance(self._engine, str):
            pass
        else:
            self._engine.dispose(close=True)
