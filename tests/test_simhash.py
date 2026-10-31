from hashbinkit.primitives import BaseSimHash


class SimHash64(BaseSimHash):
    BIT_SIZE = 64


class TestSimHash:
    def test_from_features(self):
        test_features1 = tuple(range(10))
        test_features2 = test_features1 + tuple(range(10, 16))
        sh1 = SimHash64.from_features(test_features1)
        sh2 = SimHash64.from_features(test_features2)

        assert sh1.mask().bit_length() == 64
        assert 1 < sh1.distance(sh2) < 64
        assert 0 < sh1.similarity(sh2) < 1
