__version__ = "0.2"


from transformers import AutoConfig, AutoModel

from .configuration_songgen import SongGenConfig, SongGenDecoderConfig
from .xcodec_wrapper import XCodecConfig, XCodecModel
from .modeling_songgen_mixed import (
    SongGenForCausalLM,
    SongGenMixedForConditionalGeneration,
    apply_delay_pattern_mask,
    build_delay_pattern_mask,
)

from .streamer import SongGenStreamer

from .tokenizer_xtts import VoiceBpeTokenizer

AutoConfig.register("xcodec", XCodecConfig)
AutoModel.register(XCodecConfig, XCodecModel)


