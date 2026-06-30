//! Spatial index for jigsaw piece bounds. Port of `StructureLayoutOptimizer`'s
//! `BoxOctree` — nearby-box queries instead of scanning every placed piece.

use std::mem;

use glam::IVec3;
use steel_utils::BoundingBox;

const SUBDIVIDE_THRESHOLD: usize = 10;
const MAXIMUM_DEPTH: u32 = 3;

/// Octree of axis-aligned boxes for fast intersection queries during jigsaw assembly.
#[derive(Debug, Clone)]
pub struct BoxOctree {
    boundary: BoundingBox,
    size: IVec3,
    depth: u32,
    inner_boxes: Vec<BoundingBox>,
    children: Vec<BoxOctree>,
}

impl BoxOctree {
    #[must_use]
    pub fn new(boundary: BoundingBox) -> Self {
        Self::with_depth(boundary, 0)
    }

    fn with_depth(boundary: BoundingBox, parent_depth: u32) -> Self {
        let size = IVec3::new(
            round_away_from_zero(boundary.width()),
            round_away_from_zero(boundary.height()),
            round_away_from_zero(boundary.depth()),
        );
        Self {
            boundary,
            size,
            depth: parent_depth + 1,
            inner_boxes: Vec::new(),
            children: Vec::new(),
        }
    }

    pub fn add_box(&mut self, bbox: BoundingBox) {
        if self.depth < MAXIMUM_DEPTH && self.inner_boxes.len() > SUBDIVIDE_THRESHOLD {
            self.subdivide();
        }

        if !self.children.is_empty() {
            for child in &mut self.children {
                if child.boundary_intersects(bbox) {
                    child.add_box(bbox);
                }
            }
            return;
        }

        if self.inner_boxes.contains(&bbox) {
            return;
        }
        self.inner_boxes.push(bbox);
    }

    pub fn intersects_any_box(&self, candidate: BoundingBox) -> bool {
        if !self.children.is_empty() {
            return self.children.iter().any(|child| {
                child.boundary_intersects(candidate) && child.intersects_any_box(candidate)
            });
        }
        self.inner_boxes
            .iter()
            .any(|bbox| candidate.intersects(*bbox))
    }

    fn boundary_intersects(&self, candidate: BoundingBox) -> bool {
        self.boundary.intersects(candidate)
    }

    fn subdivide(&mut self) {
        assert!(
            self.children.is_empty(),
            "BoxOctree: tried to subdivide when children already exist"
        );

        let min = self.boundary.min_corner();
        let max = self.boundary.max_corner();
        let half_x = self.size.x / 2;
        let half_y = self.size.y / 2;
        let half_z = self.size.z / 2;

        let child_bounds = [
            BoundingBox::new(
                IVec3::new(min.x, min.y, min.z),
                IVec3::new(min.x + half_x, min.y + half_y, min.z + half_z),
            ),
            BoundingBox::new(
                IVec3::new(min.x, min.y, min.z + half_z),
                IVec3::new(min.x + half_x, min.y + half_y, max.z),
            ),
            BoundingBox::new(
                IVec3::new(min.x + half_x, min.y, min.z),
                IVec3::new(max.x, min.y + half_y, min.z + half_z),
            ),
            BoundingBox::new(
                IVec3::new(min.x + half_x, min.y, min.z + half_z),
                IVec3::new(max.x, min.y + half_y, max.z),
            ),
            BoundingBox::new(
                IVec3::new(min.x, min.y + half_y, min.z),
                IVec3::new(min.x + half_x, max.y, min.z + half_z),
            ),
            BoundingBox::new(
                IVec3::new(min.x, min.y + half_y, min.z + half_z),
                IVec3::new(min.x + half_x, max.y, max.z),
            ),
            BoundingBox::new(
                IVec3::new(min.x + half_x, min.y + half_y, min.z),
                IVec3::new(max.x, max.y, min.z + half_z),
            ),
            BoundingBox::new(
                IVec3::new(min.x + half_x, min.y + half_y, min.z + half_z),
                IVec3::new(max.x, max.y, max.z),
            ),
        ];

        self.children = child_bounds
            .into_iter()
            .map(|boundary| Self::with_depth(boundary, self.depth))
            .collect();

        let inner_boxes = mem::take(&mut self.inner_boxes);
        for bbox in inner_boxes {
            for child in &mut self.children {
                if child.boundary_intersects(bbox) {
                    child.add_box(bbox);
                }
            }
        }
    }
}

const fn round_away_from_zero(value: i32) -> i32 {
    if value >= 0 { value } else { -value }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distant_boxes_do_not_intersect() {
        let boundary = BoundingBox::new(IVec3::ZERO, IVec3::new(100, 100, 100));
        let mut tree = BoxOctree::new(boundary);
        tree.add_box(BoundingBox::new(IVec3::ZERO, IVec3::new(5, 5, 5)));
        tree.add_box(BoundingBox::new(
            IVec3::new(50, 50, 50),
            IVec3::new(55, 55, 55),
        ));

        let candidate = BoundingBox::new(IVec3::new(10, 10, 10), IVec3::new(15, 15, 15));
        assert!(!tree.intersects_any_box(candidate));
    }

    #[test]
    fn nearby_boxes_intersect() {
        let boundary = BoundingBox::new(IVec3::ZERO, IVec3::new(100, 100, 100));
        let mut tree = BoxOctree::new(boundary);
        tree.add_box(BoundingBox::new(IVec3::ZERO, IVec3::new(5, 5, 5)));

        let candidate = BoundingBox::new(IVec3::new(4, 4, 4), IVec3::new(8, 8, 8));
        assert!(tree.intersects_any_box(candidate));
    }
}
