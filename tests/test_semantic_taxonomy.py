from src.semantic_taxonomy import assign_category, get_category_distribution, CATEGORIES


def test_categories_complete():
    assert set(CATEGORIES) == {'background', 'texture', 'color', 'object_part', 'scene', 'object'}


def test_background_label():
    assert assign_category('blurred background') == 'background'


def test_texture_label():
    assert assign_category('animal fur') == 'texture'
    assert assign_category('fabric texture') == 'texture'


def test_object_part_label():
    assert assign_category('muzzle') == 'object_part'
    assert assign_category('bird beak') == 'object_part'
    assert assign_category('wing') == 'object_part'


def test_scene_label():
    assert assign_category('grass texture') == 'scene'
    assert assign_category('water surface') == 'scene'


def test_imagenet_class_is_object():
    # 'patas' is an ImageNet class (patas monkey), not a texture/part/scene keyword
    assert assign_category('patas') == 'object'
    assert assign_category('disk brake') == 'object'


def test_distribution_sums():
    labels = ['blurred background', 'animal fur', 'patas', 'muzzle', 'grass texture']
    dist = get_category_distribution(labels)
    assert sum(dist.values()) == len(labels)
    assert set(dist.keys()) <= set(CATEGORIES)


def test_distribution_counts():
    labels = ['blurred background', 'blurred background', 'patas']
    dist = get_category_distribution(labels)
    assert dist['background'] == 2
    assert dist['object'] == 1


def test_color_label():
    assert assign_category('black cat') == 'color'
    assert assign_category('white dog') == 'color'
