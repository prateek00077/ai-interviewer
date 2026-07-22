"""WebRTC signaling payloads.

Only SDP crosses this boundary. Notably absent: anything identifying the
interview. That comes from the token's signed claim, so the client has no field
to tamper with.
"""

from pydantic import BaseModel, Field

# An SDP offer is a few kilobytes of text. The bound stops a candidate posting a
# megabyte of it at an endpoint that then hands it to an SDP parser.
MAX_SDP_CHARS = 64_000


class WebRTCOffer(BaseModel):
    sdp: str = Field(min_length=1, max_length=MAX_SDP_CHARS)
    type: str = Field(default="offer", pattern="^(offer|answer)$")
    # Sent back on a reconnect so the server can match an existing peer
    # connection instead of building a second one.
    pc_id: str | None = Field(default=None, max_length=128)


class WebRTCAnswer(BaseModel):
    sdp: str
    type: str
    pc_id: str | None = None
