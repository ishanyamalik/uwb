import unittest
from uwb import parse_anchor_coordinates


class UWBTest(unittest.TestCase):

    def test_parse_anchor_coordinates(self):
        anchor_coords = "[[0,0,0], [10,0,0], [0,10,0], [10,10,0]]"
        result = parse_anchor_coordinates(anchor_coords)
        expected = [[0, 0, 0], [10, 0, 0], [0, 10, 0], [10, 10, 0]]
        self.assertEqual(result.tolist(), expected)

    def test_parse_anchor_coordinates_less_than_3_anchors(self):
        anchor_coords = "[[0,0,0]]"
        with self.assertRaises(ValueError):
            parse_anchor_coordinates(anchor_coords)

    def test_parse_anchor_coordinates_less_than_3_coordinates(self):
        anchor_coords = "[[0,0], [10,0], [0,10]]"
        with self.assertRaises(ValueError):
            parse_anchor_coordinates(anchor_coords)

    def test_parse_anchor_coordinates_more_than_3_coordinates(self):
        anchor_coords = "[[0,0,0,0]]"
        with self.assertRaises(ValueError):
            parse_anchor_coordinates(anchor_coords)

    def test_parse_anchor_coordinates_not_list(self):
        anchor_coords = '{"x":"1"}'
        with self.assertRaises(ValueError):
            parse_anchor_coordinates(anchor_coords)

    def test_parse_anchor_coordinates_invalid_json(self):
        anchor_coords = "x:0"
        with self.assertRaises(ValueError):
            parse_anchor_coordinates(anchor_coords)


if __name__ == "__main__":
    unittest.main()
