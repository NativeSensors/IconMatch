import unittest

import IconMatch.rectangle as r
from IconMatch import box
from IconMatch.weighted_quick_unionUF import WeightedQuickUnionUF as uf


class TestSIFT(unittest.TestCase):
    def test_invalid_datatype(self):
        """
        No tests for sift yet
        """
        self.assertEqual(1, 1)


class TestUF(unittest.TestCase):
    """
    Test the base methods of Union Find.
    """

    def setUp(self):
        self.a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.t = uf(4, self.a[:4])
        self.u = uf(10, self.a)

    def test_count(self):
        self.assertEqual(self.t.count, 4)
        self.t.union(0, 3)

        self.assertEqual(self.t.count, 3)
        self.t.union(3, 0)

        self.assertEqual(self.t.count, 3)
        self.t.union(2, 0)

        self.assertEqual(self.t.count, 2)
        self.t.union(1, 0)

        self.assertEqual(self.t.count, 1)
        self.assertTrue(self.t.connected(1, 0))

        self.assertEqual(self.u.count, 10)

        self.u.union(0, 8)
        self.u.union(3, 5)
        self.u.union(4, 7)
        self.u.union(1, 9)
        self.assertEqual(self.u.count, 6)

    def test_find(self):
        # p, q << all things equal, p will be picked
        self.t.union(0, 3)
        self.assertEqual(self.t.find(3), 0, "Parent should be 0")

        self.t.union(3, 0)
        self.assertEqual(self.t.find(3), 0, "No change is expected")
        self.assertEqual(self.t.size[0], 2, "Parent 0 should have 2 components")

        self.t.union(2, 0)
        self.assertEqual(self.t.size[0], 3, "Parent 0 should have 3 components")

        self.t.union(1, 0)
        self.assertEqual(self.t.size[0], 4, "Parent 0 should have 4 components")

    def test_get_unions(self):
        self.u.union(0, 9)
        self.u.union(1, 8)
        self.u.union(2, 7)
        self.u.union(3, 6)
        self.u.union(4, 5)

        unions_u = self.u.get_unions()
        self.assertCountEqual([0, 1, 2, 3, 4], unions_u.keys())

        self.t.union(0, 3)
        self.t.union(2, 0)
        self.t.union(1, 0)
        unions_t = self.t.get_unions()
        self.assertCountEqual([0], unions_t.keys())


class TestRectangle(unittest.TestCase):
    """
    Test Rectangle Methods
    """

    def setUp(self):
        self.global_rect = r.Rectangle(3, 3, 9, 9)

    def test_same_intersect(self):
        """
        Tests that a rectangle overlaps with itself
        """
        a = r.Rectangle(2, 2, 8, 8)

        # the same rect should overlap
        self.assertTrue(r.Rectangle.intersect(a, a))

    def test_types_of_overlap(self):
        """
        Test different types of overlap
        """
        # top, left, bottom, right
        a = r.Rectangle(2, 2, 8, 8)
        c = r.Rectangle(4, 4, 8, 8)
        d = r.Rectangle(2, 2, 6, 6)
        e = r.Rectangle(2, 4, 8, 8)
        f = r.Rectangle(2, 2, 8, 6)

        # One rectangle overlapping with bottom right
        self.assertTrue(r.Rectangle.intersect(a, c))

        # Top left
        self.assertTrue(r.Rectangle.intersect(a, d))

        # Top right
        self.assertTrue(r.Rectangle.intersect(a, e))

        # Bottom left
        self.assertTrue(r.Rectangle.intersect(a, f))

    def test_types_of_non_overlap(self):
        """
        Test different types of non-overlap
        """
        # top, left, bottom, right
        a = r.Rectangle(2, 2, 8, 8)
        c = r.Rectangle(10, 10, 16, 16)
        d = r.Rectangle(2, 8, 8, 12)
        e = r.Rectangle(8, 2, 10, 8)
        f = r.Rectangle(8, 8, 16, 16)
        g = r.Rectangle(0, 0, 2, 2)

        self.assertFalse(r.Rectangle.intersect(a, c))
        self.assertFalse(r.Rectangle.intersect(a, d))
        self.assertFalse(r.Rectangle.intersect(a, e))
        self.assertFalse(r.Rectangle.intersect(a, f))
        self.assertFalse(r.Rectangle.intersect(a, g))

    def test_conglomeration(self):
        """
        Test that the complete bounding rectangle is returned.
        """
        # top, left, bottom, right
        a = r.Rectangle(2, 2, 8, 8)
        c = r.Rectangle(4, 4, 8, 8)
        d = r.Rectangle(2, 2, 6, 6)
        e = r.Rectangle(2, 4, 8, 8)
        f = r.Rectangle(2, 2, 8, 6)

        g = r.Rectangle(2, 2, 16, 16)
        h = r.Rectangle(15, 15, 16, 16)

        self.assertEqual(a, r.Rectangle.merge_rects([a, c, d, e, f]))

        self.assertEqual(g, r.Rectangle.merge_rects([a, h]))

    def test_contains_point(self):
        """
        Test multiple points being in and out of the global rectangle
        """

        # points that are inside, or on the fringe
        point1 = (4, 4)
        point2 = (3, 3)
        point3 = (3, 9)
        point4 = (9, 3)

        self.assertTrue(self.global_rect.contains_point(point1))
        self.assertTrue(self.global_rect.contains_point(point2))
        self.assertTrue(self.global_rect.contains_point(point3))
        self.assertTrue(self.global_rect.contains_point(point4))

        # points that are outside
        point1a = (1, 4)
        point2a = (10, 3)
        point3a = (3, 90)
        point4a = (9, -1)

        self.assertFalse(self.global_rect.contains_point(point1a))
        self.assertFalse(self.global_rect.contains_point(point2a))
        self.assertFalse(self.global_rect.contains_point(point3a))
        self.assertFalse(self.global_rect.contains_point(point4a))

    def test_distance_to_point(self):
        """
        Test multiple distances for rectangle
        """

        # points that are inside, or on the fringe
        point1 = (4, 4)
        point2 = (3, 3)
        point3 = (3, 9)
        point4 = (9, 3)

        self.assertAlmostEqual(0.0, self.global_rect.distance_to_point(point1))
        self.assertAlmostEqual(0.0, self.global_rect.distance_to_point(point2))
        self.assertAlmostEqual(0.0, self.global_rect.distance_to_point(point3))
        self.assertAlmostEqual(0.0, self.global_rect.distance_to_point(point4))

        # points that are outside
        point1a = (1, 4)
        point2a = (10, 3)
        point3a = (3, 90)
        point4a = (9, -1)

        self.assertAlmostEqual(2.0, self.global_rect.distance_to_point(point1a))
        self.assertAlmostEqual(1.0, self.global_rect.distance_to_point(point2a))
        self.assertAlmostEqual(81.0, self.global_rect.distance_to_point(point3a))
        self.assertAlmostEqual(4.0, self.global_rect.distance_to_point(point4a))


class TestBox(unittest.TestCase):
    """
    Test functionality in Box
    """

    def setUp(self):
        self.google_rectangles = [r.Rectangle(193, 279, 236, 297),
                                  r.Rectangle(255, 279, 275, 294),
                                  r.Rectangle(241, 282, 250, 294),
                                  r.Rectangle(343, 279, 353, 294),
                                  r.Rectangle(353, 279, 410, 298),
                                  r.Rectangle(316, 279, 333, 294),
                                  r.Rectangle(275, 280, 310, 294),
                                  r.Rectangle(272, 279, 274, 293),
                                  r.Rectangle(237, 279, 240, 294),
                                  r.Rectangle(391, 225, 428, 240),
                                  r.Rectangle(319, 226, 338, 237),
                                  r.Rectangle(341, 225, 387, 240),
                                  r.Rectangle(228, 226, 273, 237),
                                  r.Rectangle(178, 225, 224, 240),
                                  r.Rectangle(559, 151, 573, 171),
                                  r.Rectangle(29, 152, 43, 167),
                                  r.Rectangle(396, 44, 437, 89),
                                  r.Rectangle(334, 44, 376, 108),
                                  r.Rectangle(286, 44, 331, 89),
                                  r.Rectangle(238, 44, 283, 89),
                                  r.Rectangle(382, 21, 392, 87),
                                  r.Rectangle(382, 21, 392, 87),
                                  r.Rectangle(167, 19, 235, 89),
                                  r.Rectangle(167, 19, 235, 89),
                                  ]

        self.G_blue = r.Rectangle(14, 6, 84, 74)
        self.o_red = r.Rectangle(39, 77, 84, 122)
        self.o_yellow = r.Rectangle(39, 125, 84, 170)
        self.g_blue = r.Rectangle(39, 173, 103, 215)
        self.l_green = r.Rectangle(16, 221, 82, 231)
        self.e_red = r.Rectangle(39, 235, 84, 276)
        self.google_rectangles_small = [self.G_blue,
                                        self.o_red,
                                        self.o_yellow,
                                        self.g_blue,
                                        self.l_green,
                                        self.e_red,
                                        ]

    def test_in_rectangle_out_rectangle(self):
        point_under_capital_g_blue = (40, 45)
        point_in_o_red = (100, 63)
        point_in_o_yellow = (151, 63)
        point_in_g_blue = (196, 72)
        point_in_l_green = (226, 51)
        point_in_e_red = (259, 66)

        # inside the rectangle
        self.assertTrue(self.G_blue.contains_point(point_under_capital_g_blue))
        self.assertTrue(self.o_red.contains_point(point_in_o_red))
        self.assertTrue(self.o_yellow.contains_point(point_in_o_yellow))
        self.assertTrue(self.g_blue.contains_point(point_in_g_blue))
        self.assertTrue(self.l_green.contains_point(point_in_l_green))
        self.assertTrue(self.e_red.contains_point(point_in_e_red))

        # outside the rectangle
        self.assertFalse(self.G_blue.contains_point(point_in_o_red))
        self.assertFalse(self.o_red.contains_point(point_in_o_yellow))
        self.assertFalse(self.o_yellow.contains_point(point_in_o_red))
        self.assertFalse(self.g_blue.contains_point(point_in_o_red))
        self.assertFalse(self.l_green.contains_point(point_in_o_red))
        self.assertFalse(self.e_red.contains_point(point_in_o_red))

    def test_closest_rectangle_no_ties(self):
        point_under_capital_g_blue = (7, 95)
        point_under_o_red = (100, 92)
        point_under_o_yellow = (148, 89)
        point_under_g_blue = (193, 110)
        point_above_l_green = (226, 11)
        point_under_e_red = (259, 89)

        self.assertEqual(self.G_blue, box.closest_rectangle(self.google_rectangles_small, point_under_capital_g_blue))
        self.assertEqual(self.o_red, box.closest_rectangle(self.google_rectangles_small, point_under_o_red))
        self.assertEqual(self.o_yellow, box.closest_rectangle(self.google_rectangles_small, point_under_o_yellow))
        self.assertEqual(self.g_blue, box.closest_rectangle(self.google_rectangles_small, point_under_g_blue))
        self.assertEqual(self.l_green, box.closest_rectangle(self.google_rectangles_small, point_above_l_green))
        self.assertEqual(self.e_red, box.closest_rectangle(self.google_rectangles_small, point_under_e_red))

    def test_candidate_rectangle(self):
        # TODO: Complete this integration test.
        pass

    if __name__ == "__main__":
        unittest.main()
