import pytest

from core import BufferSpec, ConfigurationError, NetworkSpec, NodeSpec, ServerSpec
from core.network import Network

from .conftest import make_request, star


def test_ids_order_and_tree():
    deep = NodeSpec(30, (NodeSpec(20, (ServerSpec(11), ServerSpec(11, id="fast"))), ServerSpec(5)))
    network = Network(NetworkSpec((deep, NodeSpec(10, (ServerSpec(10),)))))
    assert [s.id for s in network.servers] == ["server-1", "fast", "server-3", "server-4"]
    assert [s.index for s in network.servers] == [0, 1, 2, 3]
    assert [n.id for n in network.servers[0].ancestors] == ["node-2", "node-1"]
    assert network.servers[2].parent is network.roots[0]
    assert network.servers[3].root is network.roots[1]
    assert len(network.roots[0].servers) == 3


def test_headroom():
    network = Network(star(4, servers_per_node=2, root_rate=30))
    rates = [11.0, 5.0, 0.0, 0.0]
    assert network.servers[0].parent.headroom(rates) == 6
    assert network.roots[0].headroom(rates, background=4) == 10
    assert network.roots[0].headroom(rates, background=40) == 0


def test_session_physics():
    server = Network(star(1)).servers[0]
    assert server.max_rate_now() == 0
    server.connect(make_request(1, 0, 4, level=0.5))  # capacity 50
    assert server.rate_to_reach(1.0, step_hours=1) == 25
    server.serve(11, step_hours=1)
    assert server.session.level == pytest.approx(0.72)
    server.serve(100, step_hours=1)
    assert server.session.level == 1
    assert server.rate_to_reach(0.9, step_hours=1) == 0
    assert server.disconnect().request.id == 1
    assert server.session is None


def test_buffer_limits():
    network = Network(star(1, buffer=BufferSpec(10, 4, initial_level=0.5, min_level=0.2)))
    (buffer,) = network.buffers
    assert buffer.max_release_rate(0, step_hours=1) == 3  # quantity-limited
    assert buffer.max_release_rate(0, step_hours=0.25) == 4  # rate-limited
    assert buffer.absorb(100, step_hours=1) == 4
    assert buffer.level == pytest.approx(0.9)
    assert buffer.absorb(100, step_hours=1) == pytest.approx(1)
    buffer.release(4, step_hours=1)
    assert buffer.level == pytest.approx(0.6)


@pytest.mark.parametrize(
    "build",
    [
        lambda: ServerSpec(0),
        lambda: ServerSpec(5, min_rate=6),
        lambda: NodeSpec(10, ()),
        lambda: NodeSpec(10, (NodeSpec(5, (ServerSpec(5),), buffer=BufferSpec(1, 1)),)),
        lambda: BufferSpec(10, 5, initial_level=0.1, min_level=0.2),
        lambda: NetworkSpec(()),
    ],
)
def test_invalid_specs(build):
    with pytest.raises(ConfigurationError):
        build()


def test_numpy_numbers_are_accepted():
    import numpy as np

    node = NodeSpec(np.int64(15), (ServerSpec(np.float32(3.7)),))
    assert node.max_rate == 15


def test_buffer_round_trip_losses():
    network = Network(star(1, buffer=BufferSpec(10, 100, efficiency=0.81)))  # one way 0.9
    (buffer,) = network.buffers
    drawn = buffer.absorb(100, step_hours=1)
    assert drawn == pytest.approx(10 / 0.9)  # 11.1 kWh drawn to store 10 kWh
    assert buffer.level == pytest.approx(1)
    assert buffer.max_release_rate(0, step_hours=1) == pytest.approx(9, abs=1e-3)
    buffer.release(9, step_hours=1)
    assert buffer.level == pytest.approx(0, abs=1e-3)  # 9 delivered of 11.1 drawn: 81 %
