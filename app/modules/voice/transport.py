"""SmallWebRTCConnection setup and signaling glue.

WebRTC rather than a raw audio WebSocket, because the browser's jitter buffer,
packet-loss concealment and echo cancellation come with it. On a candidate's
home wifi those are the difference between a usable call and one where every
third word drops.

DEPLOYMENT DEBT, stated plainly: only host ICE candidates are configured, so
this works browser-to-server on the same machine or the same LAN. A real
deployment needs STUN for NAT traversal and TURN for the candidates whose
network blocks direct paths -- roughly one in ten in practice. Both are
configuration on the connection below rather than code changes here.
"""

from __future__ import annotations

import structlog
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import IceServer, SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from app.core.config import settings
from app.modules.voice.nvidia import stt as stt_service
from app.modules.voice.nvidia import tts as tts_service

log = structlog.get_logger(__name__)


def ice_servers() -> list[IceServer]:
    """STUN servers, if any are configured.

    Empty by default: an empty list means host candidates only, which is
    correct for local development and insufficient for production.
    """
    return [IceServer(urls=url) for url in settings.webrtc_stun_urls]


def create_connection() -> SmallWebRTCConnection:
    return SmallWebRTCConnection(ice_servers=ice_servers())


def transport_params() -> TransportParams:
    """Audio in and out, no video.

    The sample rates deliberately differ: 16k in is what Riva ASR wants, 44.1k
    out is what Magpie synthesises natively. Forcing them to match would add a
    resample on one side for no benefit.
    """
    return TransportParams(
        audio_in_enabled=True,
        audio_in_sample_rate=stt_service.SAMPLE_RATE,
        audio_in_channels=1,
        audio_out_enabled=True,
        audio_out_sample_rate=tts_service.SAMPLE_RATE,
        audio_out_channels=1,
        # Video is off: proctoring frames go to S3 over HTTP, not through the
        # voice pipeline, so there is no reason to carry a video track.
        video_in_enabled=False,
        video_out_enabled=False,
    )


def build(connection: SmallWebRTCConnection) -> SmallWebRTCTransport:
    log.info("voice.transport_configured", stun=len(settings.webrtc_stun_urls))
    return SmallWebRTCTransport(webrtc_connection=connection, params=transport_params())
