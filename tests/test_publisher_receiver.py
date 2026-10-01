import json
import socket

from ctrack.publisher.udp import UdpPublisher
from ctrack.types import Message, Pose, TrackState, Velocity
from receiver import StreamStats


def _msg(seq, t=0.0, state=TrackState.TRACKING):
    return Message(seq, state, t, t + 0.01, Pose(1, 2, 0.1), Velocity(3, 4, 0.0), "px", "image", 0.9)


def _listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(1.0)
    return sock, sock.getsockname()[1]


def test_udp_roundtrip():
    sock, port = _listener()
    pub = UdpPublisher("127.0.0.1", port)
    assert pub.publish(_msg(5))
    data = json.loads(sock.recv(65535))
    assert data["seq"] == 5 and data["state"] == "TRACKING" and data["pose"]["x"] == 1
    pub.close()
    sock.close()


def test_rate_limit_drops_messages():
    sock, port = _listener()
    pub = UdpPublisher("127.0.0.1", port, rate_hz=10)
    sent = [pub.publish(_msg(i, t=i * 0.033)) for i in range(10)]  # 30 fps in, 10 Hz out
    assert [i for i, ok in enumerate(sent) if ok] == [0, 4, 8]  # t = 0, 0.132, 0.264: >= 100 ms apart
    pub.close()
    sock.close()


def test_receiver_counts_seq_gaps():
    s = StreamStats()
    for seq in [0, 1, 2, 5, 6, 6, 7]:
        s.add({"seq": seq, "state": "TRACKING", "t_exposure": 0.0}, t_recv=seq * 0.033)
    assert s.lost == 2
    assert s.reordered == 1
    assert s.received == 6
