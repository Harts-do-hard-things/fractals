from fractals import HeighwayDragon


def test_smoke_import_and_iterate():
    dragon = HeighwayDragon()
    dragon.iterate(1)
    assert len(dragon.S) > 0
