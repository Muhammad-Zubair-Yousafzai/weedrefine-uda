from weedrefine.data.splits import domain_name, split_stems


def test_split_sizes_disjoint_complete():
    stems = [f"im_{i:03d}" for i in range(125)]
    train, val = split_stems(stems, 0.2, seed=0)
    assert len(val) == 25 and len(train) == 100
    assert not set(train) & set(val)
    assert set(train) | set(val) == set(stems)


def test_split_deterministic_and_seed_dependent():
    stems = [f"im_{i:03d}" for i in range(125)]
    assert split_stems(stems, 0.2, 1) == split_stems(list(reversed(stems)), 0.2, 1)
    assert split_stems(stems, 0.2, 0)[1] != split_stems(stems, 0.2, 1)[1]


def test_domain_name():
    assert domain_name("Bipbip", "Haricot", "2021") == "bipbip_haricot_2021"
