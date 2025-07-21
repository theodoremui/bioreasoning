import tempfile as temp
import os
import uuid
from dotenv import load_dotenv
import logging
from contextlib import asynccontextmanager

from pydub import AudioSegment
from elevenlabs import AsyncElevenLabs
from llama_index.core.llms.structured_llm import StructuredLLM
from typing_extensions import Self
from typing import List, Literal, Optional, AsyncIterator
from pydantic import BaseModel, ConfigDict, model_validator, Field
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAIResponses

logger = logging.getLogger(__name__)


class ConversationTurn(BaseModel):
    speaker: Literal["speaker1", "speaker2"] = Field(
        description="The person who is speaking",
    )
    content: str = Field(
        description="The content of the speech",
    )


class MultiTurnConversation(BaseModel):
    conversation: List[ConversationTurn] = Field(
        description="List of conversation turns. Conversation must start with speaker1, and continue with an alternance of speaker1 and speaker2",
        min_length=3,
        max_length=50,
        examples=[
            [
                ConversationTurn(speaker="speaker1", content="Hello, who are you?"),
                ConversationTurn(
                    speaker="speaker2", content="I am very well, how about you?"
                ),
                ConversationTurn(speaker="speaker1", content="I am well too, thanks!"),
            ]
        ],
    )

    @model_validator(mode="after")
    def validate_conversation(self) -> Self:
        speakers = [turn.speaker for turn in self.conversation]
        if speakers[0] != "speaker1":
            raise ValueError("Conversation must start with speaker1")
        for i, speaker in enumerate(speakers):
            if i % 2 == 0 and speaker != "speaker1":
                raise ValueError(
                    "Conversation must be an alternance between speaker1 and speaker2"
                )
            elif i % 2 != 0 and speaker != "speaker2":
                raise ValueError(
                    "Conversation must be an alternance between speaker1 and speaker2"
                )
            continue
        return self


class VoiceConfig(BaseModel):
    """Configuration for voice settings"""

    speaker1_voice_id: str = Field(
        default="nPczCjzI2devNBz1zQrb", description="Voice ID for speaker 1"
    )
    speaker2_voice_id: str = Field(
        default="Xb7hH8MSUJpSbSDYk0k2", description="Voice ID for speaker 2"
    )
    model_id: str = Field(
        default="eleven_turbo_v2_5", description="ElevenLabs model ID"
    )
    output_format: str = Field(
        default="mp3_22050_32", description="Audio output format"
    )


class AudioQuality(BaseModel):
    """Configuration for audio quality settings"""

    bitrate: str = Field(default="320k", description="Audio bitrate")
    quality_params: List[str] = Field(
        default=["-q:a", "0"], description="Additional quality parameters"
    )


class PodcastConfig(BaseModel):
    """Configuration for podcast generation"""

    # Basic style options
    style: Literal["conversational", "interview", "debate", "educational"] = Field(
        default="conversational",
        description="The style of the conversation",
    )
    tone: Literal["friendly", "professional", "casual", "energetic"] = Field(
        default="friendly",
        description="The tone of the conversation",
    )

    # Content focus
    focus_topics: Optional[List[str]] = Field(
        default=None, description="Specific topics to focus on during the conversation"
    )

    # Audience targeting
    target_audience: Literal[
        "general", "technical", "business", "expert", "beginner"
    ] = Field(
        default="general",
        description="The target audience for the conversation",
    )

    # Custom user prompt
    custom_prompt: Optional[str] = Field(
        default=None, description="Additional instructions for the conversation"
    )

    # Speaker customization
    speaker1_role: str = Field(
        default="host", description="The role of the first speaker"
    )
    speaker2_role: str = Field(
        default="guest", description="The role of the second speaker"
    )

    # Voice and audio configuration
    voice_config: VoiceConfig = Field(
        default_factory=VoiceConfig, description="Voice configuration for speakers"
    )
    audio_quality: AudioQuality = Field(
        default_factory=AudioQuality, description="Audio quality settings"
    )


class PodcastGeneratorError(Exception):
    """Base exception for podcast generator errors"""

    pass


class AudioGenerationError(PodcastGeneratorError):
    """Raised when audio generation fails"""

    pass


class ConversationGenerationError(PodcastGeneratorError):
    """Raised when conversation generation fails"""

    pass


class PodcastGenerator(BaseModel):
    llm: StructuredLLM
    client: AsyncElevenLabs

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_podcast(self) -> Self:
        try:
            assert self.llm.output_cls == MultiTurnConversation
        except AssertionError:
            raise ValueError(
                f"The output class of the structured LLM must be {MultiTurnConversation.__qualname__}, your LLM has output class: {self.llm.output_cls.__qualname__}"
            )
        return self

    def _build_conversation_prompt(
        self, file_transcript: str, config: PodcastConfig
    ) -> str:
        """Build a customized prompt based on the configuration"""

        # Base prompt with style and tone
        prompt = f"""Create a {config.style} podcast conversation with two speakers from this transcript.

        CONVERSATION STYLE: {config.style}
        TONE: {config.tone}
        TARGET AUDIENCE: {config.target_audience}

        SPEAKER ROLES:
        - Speaker 1: {config.speaker1_role}
        - Speaker 2: {config.speaker2_role}
        """

        if config.focus_topics:
            prompt += "\nFOCUS TOPICS: Make sure to discuss these topics in detail:\n"
            for topic in config.focus_topics:
                prompt += f"- {topic}\n"

        # Add audience-specific instructions
        audience_instructions = {
            "technical": "Use technical terminology appropriately and dive deep into technical details.",
            "beginner": "Explain concepts clearly and avoid jargon. Define technical terms when used.",
            "expert": "Assume advanced knowledge and discuss nuanced aspects and implications.",
            "business": "Focus on practical applications, ROI, and strategic implications.",
            "general": "Balance accessibility with depth, explaining key concepts clearly.",
        }

        prompt += (
            f"\nAUDIENCE APPROACH: {audience_instructions[config.target_audience]}\n"
        )

        # Add custom prompt if provided
        if config.custom_prompt:
            prompt += f"\nADDITIONAL INSTRUCTIONS: {config.custom_prompt}\n"

        # Add the source material
        prompt += f"\nSOURCE MATERIAL:\n'''\n{file_transcript}\n'''\n"

        # Add final instructions
        prompt += """
        IMPORTANT: Create an engaging, natural conversation that flows well between the two speakers.
        The conversation should feel authentic and provide value to the target audience.
        """

        return prompt

    async def _conversation_script(
        self, file_transcript: str, config: PodcastConfig
    ) -> MultiTurnConversation:
        """Generate conversation script with customization"""
        logger.info("Generating conversation script...")
        prompt = self._build_conversation_prompt(file_transcript, config)

        response = await self.llm.achat(
            messages=[
                ChatMessage(
                    role="user",
                    content=prompt,
                )
            ]
        )

        conversation = MultiTurnConversation.model_validate_json(
            response.message.content
        )
        logger.info(
            f"Generated conversation with {len(conversation.conversation)} turns"
        )
        return conversation

    @asynccontextmanager
    async def _cleanup_files(self, files: List[str]) -> AsyncIterator[None]:
        """Context manager to ensure temporary files are cleaned up"""
        try:
            yield
        finally:
            for file_path in files:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        logger.debug(f"Cleaned up temporary file: {file_path}")
                except OSError as e:
                    logger.warning(f"Failed to clean up file {file_path}: {str(e)}")

    async def _generate_speech_file(
        self, text: str, voice_id: str, config: PodcastConfig
    ) -> str:
        """Generate speech file for a single turn"""
        try:
            speech_iterator = self.client.text_to_speech.convert(
                voice_id=voice_id,
                text=text,
                output_format=config.voice_config.output_format,
                model_id=config.voice_config.model_id,
            )

            temp_file = temp.NamedTemporaryFile(
                suffix=".mp3", delete=False, delete_on_close=False
            )

            with open(temp_file.name, "wb") as f:
                async for chunk in speech_iterator:
                    if chunk:
                        f.write(chunk)

            return temp_file.name
        except Exception as e:
            logger.error(f"Failed to generate speech for text: {text[:50]}")
            raise AudioGenerationError(
                f"Failed to generate speech for text: {text[:50]}"
            ) from e

    async def _conversation_audio(
        self, conversation: MultiTurnConversation, config: PodcastConfig
    ) -> str:
        """Generate audio for the conversation"""
        files: List[str] = []

        async with self._cleanup_files(files):
            try:
                logger.info("Generating audio for conversation")

                for i, turn in enumerate(conversation.conversation):
                    voice_id = (
                        config.voice_config.speaker1_voice_id
                        if turn.speaker == "speaker1"
                        else config.voice_config.speaker2_voice_id
                    )
                    file_path = await self._generate_speech_file(
                        turn.content, voice_id, config
                    )
                    files.append(file_path)

                logger.info("Combining audio files...")
                output_path = f"conversation_{str(uuid.uuid4())}.mp3"
                combined_audio: AudioSegment = AudioSegment.empty()

                for file_path in files:
                    audio = AudioSegment.from_file(file_path)
                    combined_audio += audio

                combined_audio.export(
                    output_path,
                    format="mp3",
                    bitrate=config.audio_quality.bitrate,
                    parameters=config.audio_quality.quality_params,
                )

                logger.info(f"Successfully created podcast audio: {output_path}")
                return output_path
            except Exception as e:
                logger.error(f"Failed to generate conversation audio: {str(e)}")
                raise AudioGenerationError(
                    f"Failed to generate conversation audio: {str(e)}"
                ) from e

    async def create_conversation(
        self, file_transcript: str, config: Optional[PodcastConfig] = None
    ):
        """Main method to create a customized podcast conversation"""
        if config is None:
            config = PodcastConfig()

        try:
            logger.info("Starting podcast generation...")

            conversation = await self._conversation_script(
                file_transcript=file_transcript, config=config
            )
            podcast_file = await self._conversation_audio(
                conversation=conversation, config=config
            )

            logger.info("Podcast generation completed successfully")
            return podcast_file

        except (ConversationGenerationError, AudioGenerationError) as e:
            logger.error(f"Failed to generate podcast: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in podcast generation: {str(e)}")
            raise PodcastGeneratorError(
                f"Unexpected error in podcast generation: {str(e)}"
            ) from e


load_dotenv()

PODCAST_GEN: Optional[PodcastGenerator]

if os.getenv("ELEVENLABS_API_KEY", None) and os.getenv("OPENAI_API_KEY", None):
    SLLM = OpenAIResponses(
        model="gpt-4.1", api_key=os.getenv("OPENAI_API_KEY")
    ).as_structured_llm(MultiTurnConversation)
    EL_CLIENT = AsyncElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
    PODCAST_GEN = PodcastGenerator(llm=SLLM, client=EL_CLIENT)
else:
    logger.warning("Missing API keys - PODCAST_GEN not initialized")
    PODCAST_GEN = None
