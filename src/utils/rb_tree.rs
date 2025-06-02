use alloc::{boxed::Box, vec::Vec};
use core::mem::MaybeUninit;
use ordered_float::{FloatCore, OrderedFloat};

/// Red-Black tree node colors used to maintain tree balance properties.
///
/// Red-Black trees maintain balance by ensuring:
/// - Red nodes have black children
/// - All paths from root to leaves have equal black node counts
#[derive(Debug, Clone, Copy, PartialEq)]
enum Color {
    /// Red node - must have black children, cannot be adjacent to other red nodes
    Red,
    /// Black node - can have children of any color, contributes to black height
    Black,
}

/// A node in the Red-Black tree containing data and structural information.
///
/// Each node stores its value, duplicate count, tree relationships, color for balancing,
/// and subtree statistics for efficient order statistics operations.
#[derive(Debug, Clone)]
struct Node<T> {
    /// The stored value wrapped in OrderedFloat for consistent comparison including NaN
    value: OrderedFloat<T>,

    /// Number of duplicate values stored in this node (supports multiset behavior)
    count: u32,

    /// Index of parent node in the nodes array (nil if this is root)
    parent: usize,

    /// Index of left child node in the nodes array (nil if no left child)
    left: usize,

    /// Index of right child node in the nodes array (nil if no right child)  
    right: usize,

    /// Color of this node (Red or Black) used for Red-Black tree balancing
    color: Color,

    /// Total count of elements in this node's subtree (including duplicates)
    /// Used for efficient quantile and order statistic calculations
    subtree_count: usize,
}

/// A Red-Black tree implementation optimized for quantile calculations and sliding windows.
///
/// This tree provides O(log n) insertions, deletions, and quantile queries with support
/// for duplicate values. Memory is pre-allocated and reused via a free list for
/// consistent performance in real-time applications.
///
/// Key features:
/// - Fixed capacity with no dynamic allocation after initialization
/// - Efficient quantile/percentile calculations via subtree counts
/// - Duplicate value support (multiset behavior)
/// - Memory reuse through internal free list management
#[derive(Debug)]
pub struct RbTree<T> {
    /// Pre-allocated array of potentially uninitialized nodes
    /// Uses MaybeUninit for safe handling of uninitialized memory
    nodes: Box<[MaybeUninit<Node<T>>]>,

    /// Stack of available node indices for allocation
    /// Acts as a LIFO stack where free_top points to the next available slot
    free_list: Box<[usize]>,

    /// Index pointing to the next free slot in free_list (stack top)
    /// When free_top == 0, no free nodes are available
    free_top: usize,

    /// Number of unique values currently stored in the tree
    /// Does not count duplicates (distinct elements only)
    len: usize,

    /// Total number of elements including all duplicates
    /// Used for quantile calculations and capacity tracking
    total_count: usize,

    /// Maximum number of nodes this tree can hold
    /// Fixed at initialization, determines size of nodes and free_list arrays
    capacity: usize,

    /// Index of the root node in the nodes array
    /// Equal to nil when tree is empty
    root: usize,

    /// Sentinel value representing null/empty nodes
    /// Typically set to capacity (one past the last valid index)
    nil: usize,
}

#[allow(dead_code)]
impl<T: FloatCore + Copy> RbTree<T> {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "Capacity must be greater than 0");
        assert!(
            capacity <= usize::MAX / 2,
            "Capacity too large - risk of index overflow"
        );

        let nodes = Box::new_uninit_slice(capacity);
        let free_list = (0..capacity).collect::<Vec<_>>().into_boxed_slice();

        Self {
            nodes,
            free_list,
            free_top: capacity,
            len: 0,
            total_count: 0,
            capacity,
            root: capacity,
            nil: capacity,
        }
    }

    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub const fn total_count(&self) -> usize {
        self.total_count
    }

    #[inline]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub const fn remaining_capacity(&self) -> usize {
        self.capacity - self.len
    }

    pub fn insert(&mut self, value: T) -> Option<usize> {
        let ordered_value = OrderedFloat(value);

        if let Some(existing_idx) = self.find_node(value) {
            self.increment_count(existing_idx);
            return Some(existing_idx);
        }

        let new_idx = self.allocate_node()?;
        let parent_idx = self.find_insertion_parent(ordered_value);

        self.create_node(new_idx, ordered_value, parent_idx);
        self.link_to_parent(new_idx, parent_idx, ordered_value);

        self.len += 1;
        self.total_count += 1;

        self.update_subtree_counts_to_root(new_idx);
        self.fix_insertion_violations(new_idx);

        #[cfg(debug_assertions)]
        debug_assert!(
            self.verify_rb_invariants(),
            "RB tree invariants violated after insertion"
        );

        Some(new_idx)
    }

    pub fn remove(&mut self, value: T) -> Option<T> {
        let node_idx = self.find_node(value)?;

        if self.node_at(node_idx).count > 1 {
            self.decrement_count(node_idx);
        } else {
            self.delete_node(node_idx);
            self.len -= 1;
            self.total_count -= 1;
        }

        #[cfg(debug_assertions)]
        debug_assert!(
            self.verify_rb_invariants(),
            "RB tree invariants violated after removal"
        );

        Some(value)
    }

    pub fn min(&self) -> Option<T> {
        if self.root == self.nil {
            return None;
        }
        let min_node = self.find_minimum(self.root);
        Some(self.node_at(min_node).value.into_inner())
    }

    pub fn max(&self) -> Option<T> {
        if self.root == self.nil {
            return None;
        }
        let max_node = self.find_maximum(self.root);
        Some(self.node_at(max_node).value.into_inner())
    }

    pub fn quantile(&self, q: f64) -> Option<T> {
        if self.total_count == 0 {
            return None;
        }

        let clamped_q = q.clamp(0.0, 1.0);
        let target_index = if clamped_q >= 1.0 {
            self.total_count - 1
        } else if clamped_q <= 0.0 {
            0
        } else {
            let exact_position = clamped_q * (self.total_count - 1) as f64;
            exact_position.floor() as usize
        };

        self.find_kth_element(target_index)
    }

    pub fn percentile(&self, p: f64) -> Option<T> {
        self.quantile(p / 100.0)
    }

    pub fn median(&self) -> Option<T> {
        self.quantile(0.5)
    }

    pub fn reset(&mut self) {
        self.len = 0;
        self.total_count = 0;
        self.root = self.nil;
        for i in 0..self.capacity {
            self.free_list[i] = i;
        }
        self.free_top = self.capacity;
    }

    #[inline]
    const fn allocate_node(&mut self) -> Option<usize> {
        if self.free_top == 0 {
            None
        } else {
            self.free_top -= 1;
            Some(self.free_list[self.free_top])
        }
    }

    #[inline]
    const fn deallocate_node(&mut self, node_idx: usize) {
        debug_assert!(node_idx < self.capacity);
        self.free_list[self.free_top] = node_idx;
        self.free_top += 1;
    }

    #[inline]
    const fn node_at(&self, idx: usize) -> &Node<T> {
        debug_assert!(idx < self.capacity);
        unsafe { self.nodes[idx].assume_init_ref() }
    }

    #[inline]
    const fn node_at_mut(&mut self, idx: usize) -> &mut Node<T> {
        debug_assert!(idx < self.capacity);
        unsafe { self.nodes[idx].assume_init_mut() }
    }

    const fn create_node(&mut self, idx: usize, value: OrderedFloat<T>, parent: usize) {
        let node = Node {
            value,
            count: 1,
            parent,
            left: self.nil,
            right: self.nil,
            color: Color::Red,
            subtree_count: 1,
        };
        self.nodes[idx].write(node);
    }

    fn link_to_parent(&mut self, node_idx: usize, parent_idx: usize, value: OrderedFloat<T>) {
        if parent_idx == self.nil {
            self.root = node_idx;
        } else if value < self.node_at(parent_idx).value {
            self.node_at_mut(parent_idx).left = node_idx;
        } else {
            self.node_at_mut(parent_idx).right = node_idx;
        }
    }

    fn find_node(&self, value: T) -> Option<usize> {
        let value = OrderedFloat(value);
        let mut current = self.root;

        while current != self.nil {
            let node = self.node_at(current);
            match value.cmp(&node.value) {
                core::cmp::Ordering::Equal => return Some(current),
                core::cmp::Ordering::Less => current = node.left,
                core::cmp::Ordering::Greater => current = node.right,
            }
        }
        None
    }

    fn find_insertion_parent(&self, value: OrderedFloat<T>) -> usize {
        if self.root == self.nil {
            return self.nil;
        }

        let mut current = self.root;
        let mut parent = self.nil;

        while current != self.nil {
            parent = current;
            let node = self.node_at(current);
            current = if value < node.value {
                node.left
            } else {
                node.right
            };
        }
        parent
    }

    const fn find_minimum(&self, mut node: usize) -> usize {
        while node != self.nil {
            let left = self.node_at(node).left;
            if left == self.nil {
                break;
            }
            node = left;
        }
        node
    }

    const fn find_maximum(&self, mut node: usize) -> usize {
        while node != self.nil {
            let right = self.node_at(node).right;
            if right == self.nil {
                break;
            }
            node = right;
        }
        node
    }

    const fn increment_count(&mut self, node_idx: usize) {
        self.node_at_mut(node_idx).count += 1;
        self.total_count += 1;
        self.update_subtree_counts_to_root(node_idx);
    }

    const fn decrement_count(&mut self, node_idx: usize) {
        self.node_at_mut(node_idx).count -= 1;
        self.total_count -= 1;
        self.update_subtree_counts_to_root(node_idx);
    }

    const fn update_subtree_counts_to_root(&mut self, mut node: usize) {
        while node != self.nil {
            self.recalculate_subtree_count(node);
            node = self.node_at(node).parent;
        }
    }

    const fn recalculate_subtree_count(&mut self, node_idx: usize) {
        if node_idx == self.nil {
            return;
        }

        let node = self.node_at(node_idx);
        let left_count = self.get_subtree_count(node.left);
        let right_count = self.get_subtree_count(node.right);

        let total = (node.count as usize)
            .saturating_add(left_count)
            .saturating_add(right_count);

        self.node_at_mut(node_idx).subtree_count = total;
    }

    const fn get_subtree_count(&self, node_idx: usize) -> usize {
        if node_idx == self.nil {
            0
        } else {
            self.node_at(node_idx).subtree_count
        }
    }

    fn find_kth_element(&self, k: usize) -> Option<T> {
        if k >= self.total_count || self.root == self.nil {
            return None;
        }

        let mut current = self.root;
        let mut remaining_rank = k;

        loop {
            if current == self.nil {
                return None;
            }

            let node = self.node_at(current);
            let left_count = self.get_subtree_count(node.left);

            if remaining_rank < left_count {
                current = node.left;
            } else if remaining_rank < left_count + node.count as usize {
                return Some(node.value.into_inner());
            } else {
                remaining_rank -= left_count + node.count as usize;
                current = node.right;
            }
        }
    }

    #[inline]
    const fn get_color(&self, node_idx: usize) -> Color {
        if node_idx == self.nil {
            Color::Black
        } else {
            self.node_at(node_idx).color
        }
    }

    #[inline]
    const fn set_color(&mut self, node_idx: usize, color: Color) {
        if node_idx != self.nil {
            self.node_at_mut(node_idx).color = color;
        }
    }

    #[inline]
    fn is_red(&self, node_idx: usize) -> bool {
        self.get_color(node_idx) == Color::Red
    }

    #[inline]
    fn is_black(&self, node_idx: usize) -> bool {
        self.get_color(node_idx) == Color::Black
    }

    const fn rotate_left(&mut self, x: usize) {
        if x == self.nil {
            return;
        }

        let y = self.node_at(x).right;
        if y == self.nil {
            return;
        }

        let y_left = self.node_at(y).left;
        self.node_at_mut(x).right = y_left;
        if y_left != self.nil {
            self.node_at_mut(y_left).parent = x;
        }

        let x_parent = self.node_at(x).parent;
        self.node_at_mut(y).parent = x_parent;

        if x_parent == self.nil {
            self.root = y;
        } else if x == self.node_at(x_parent).left {
            self.node_at_mut(x_parent).left = y;
        } else {
            self.node_at_mut(x_parent).right = y;
        }

        self.node_at_mut(y).left = x;
        self.node_at_mut(x).parent = y;

        self.recalculate_subtree_count(x);
        self.recalculate_subtree_count(y);
    }

    const fn rotate_right(&mut self, y: usize) {
        if y == self.nil {
            return;
        }

        let x = self.node_at(y).left;
        if x == self.nil {
            return;
        }

        let x_right = self.node_at(x).right;
        self.node_at_mut(y).left = x_right;
        if x_right != self.nil {
            self.node_at_mut(x_right).parent = y;
        }

        let y_parent = self.node_at(y).parent;
        self.node_at_mut(x).parent = y_parent;

        if y_parent == self.nil {
            self.root = x;
        } else if y == self.node_at(y_parent).left {
            self.node_at_mut(y_parent).left = x;
        } else {
            self.node_at_mut(y_parent).right = x;
        }

        self.node_at_mut(x).right = y;
        self.node_at_mut(y).parent = x;

        self.recalculate_subtree_count(y);
        self.recalculate_subtree_count(x);
    }

    fn fix_insertion_violations(&mut self, mut node: usize) {
        while node != self.root && self.is_red(self.get_parent(node)) {
            let parent = self.get_parent(node);
            let grandparent = self.get_parent(parent);

            if parent == self.get_left(grandparent) {
                let uncle = self.get_right(grandparent);

                if self.is_red(uncle) {
                    self.set_color(parent, Color::Black);
                    self.set_color(uncle, Color::Black);
                    self.set_color(grandparent, Color::Red);
                    node = grandparent;
                } else {
                    if node == self.get_right(parent) {
                        node = parent;
                        self.rotate_left(node);
                    }
                    let updated_parent = self.get_parent(node);
                    let updated_grandparent = self.get_parent(updated_parent);
                    self.set_color(updated_parent, Color::Black);
                    self.set_color(updated_grandparent, Color::Red);
                    self.rotate_right(updated_grandparent);
                }
            } else {
                let uncle = self.get_left(grandparent);

                if self.is_red(uncle) {
                    self.set_color(parent, Color::Black);
                    self.set_color(uncle, Color::Black);
                    self.set_color(grandparent, Color::Red);
                    node = grandparent;
                } else {
                    if node == self.get_left(parent) {
                        node = parent;
                        self.rotate_right(node);
                    }
                    let updated_parent = self.get_parent(node);
                    let updated_grandparent = self.get_parent(updated_parent);
                    self.set_color(updated_parent, Color::Black);
                    self.set_color(updated_grandparent, Color::Red);
                    self.rotate_left(updated_grandparent);
                }
            }
        }
        self.set_color(self.root, Color::Black);
    }

    fn delete_node(&mut self, node_to_delete: usize) {
        let (original_color, fixup_node, fixup_parent) = self.perform_deletion(node_to_delete);

        if original_color == Color::Black {
            self.fix_deletion_violations(fixup_node, fixup_parent);
        }

        if fixup_parent != self.nil {
            self.update_subtree_counts_to_root(fixup_parent);
        } else if self.root != self.nil {
            self.update_subtree_counts_to_root(self.root);
        }

        self.deallocate_node(node_to_delete);
    }

    const fn perform_deletion(&mut self, node: usize) -> (Color, usize, usize) {
        let original_color = self.get_color(node);
        let fixup_node;
        let mut fixup_parent;

        let left = self.node_at(node).left;
        let right = self.node_at(node).right;

        if left == self.nil {
            fixup_node = right;
            fixup_parent = self.node_at(node).parent;
            self.transplant(node, right);
        } else if right == self.nil {
            fixup_node = left;
            fixup_parent = self.node_at(node).parent;
            self.transplant(node, left);
        } else {
            let successor = self.find_minimum(right);
            let successor_color = self.get_color(successor);
            fixup_node = self.node_at(successor).right;

            if self.node_at(successor).parent == node {
                fixup_parent = successor;
            } else {
                fixup_parent = self.node_at(successor).parent;
                self.transplant(successor, self.node_at(successor).right);
                self.node_at_mut(successor).right = right;
                self.node_at_mut(right).parent = successor;
                // Update fixup_parent since tree structure changed
                if fixup_node != self.nil {
                    fixup_parent = self.node_at(fixup_node).parent;
                }
            }

            self.transplant(node, successor);
            self.node_at_mut(successor).left = left;
            self.node_at_mut(left).parent = successor;
            self.node_at_mut(successor).color = original_color;

            return (successor_color, fixup_node, fixup_parent);
        }

        (original_color, fixup_node, fixup_parent)
    }

    const fn transplant(&mut self, old_node: usize, new_node: usize) {
        let parent = self.node_at(old_node).parent;

        if parent == self.nil {
            self.root = new_node;
        } else if old_node == self.node_at(parent).left {
            self.node_at_mut(parent).left = new_node;
        } else {
            self.node_at_mut(parent).right = new_node;
        }

        if new_node != self.nil {
            self.node_at_mut(new_node).parent = parent;
        }
    }

    fn fix_deletion_violations(&mut self, mut fixup_node: usize, mut fixup_parent: usize) {
        while fixup_node != self.root && self.is_black(fixup_node) {
            if fixup_node != self.nil {
                fixup_parent = self.node_at(fixup_node).parent;
            }

            if fixup_parent == self.nil {
                break;
            }

            if fixup_node == self.get_left(fixup_parent) {
                let mut sibling = self.get_right(fixup_parent);

                if self.is_red(sibling) {
                    self.set_color(sibling, Color::Black);
                    self.set_color(fixup_parent, Color::Red);
                    self.rotate_left(fixup_parent);
                    if fixup_node != self.nil {
                        fixup_parent = self.node_at(fixup_node).parent;
                    }
                    sibling = self.get_right(fixup_parent);
                }

                if self.is_black(self.get_left(sibling)) && self.is_black(self.get_right(sibling)) {
                    self.set_color(sibling, Color::Red);
                    fixup_node = fixup_parent;
                } else {
                    if self.is_black(self.get_right(sibling)) {
                        self.set_color(self.get_left(sibling), Color::Black);
                        self.set_color(sibling, Color::Red);
                        self.rotate_right(sibling);
                        if fixup_node != self.nil {
                            fixup_parent = self.node_at(fixup_node).parent;
                        }
                        sibling = self.get_right(fixup_parent);
                    }

                    self.set_color(sibling, self.get_color(fixup_parent));
                    self.set_color(fixup_parent, Color::Black);
                    self.set_color(self.get_right(sibling), Color::Black);
                    self.rotate_left(fixup_parent);
                    fixup_node = self.root;
                }
            } else {
                let mut sibling = self.get_left(fixup_parent);

                if self.is_red(sibling) {
                    self.set_color(sibling, Color::Black);
                    self.set_color(fixup_parent, Color::Red);
                    self.rotate_right(fixup_parent);

                    if fixup_node != self.nil {
                        fixup_parent = self.node_at(fixup_node).parent;
                    }
                    sibling = self.get_left(fixup_parent);
                }

                if self.is_black(self.get_right(sibling)) && self.is_black(self.get_left(sibling)) {
                    self.set_color(sibling, Color::Red);
                    fixup_node = fixup_parent;
                } else {
                    if self.is_black(self.get_left(sibling)) {
                        self.set_color(self.get_right(sibling), Color::Black);
                        self.set_color(sibling, Color::Red);
                        self.rotate_left(sibling);
                        if fixup_node != self.nil {
                            fixup_parent = self.node_at(fixup_node).parent;
                        }
                        sibling = self.get_left(fixup_parent);
                    }

                    self.set_color(sibling, self.get_color(fixup_parent));
                    self.set_color(fixup_parent, Color::Black);
                    self.set_color(self.get_left(sibling), Color::Black);
                    self.rotate_right(fixup_parent);
                    fixup_node = self.root;
                }
            }
        }

        self.set_color(fixup_node, Color::Black);
        if self.root != self.nil {
            self.set_color(self.root, Color::Black);
        }
    }

    #[inline]
    const fn get_parent(&self, node: usize) -> usize {
        if node == self.nil {
            self.nil
        } else {
            self.node_at(node).parent
        }
    }

    #[inline]
    const fn get_left(&self, node: usize) -> usize {
        if node == self.nil {
            self.nil
        } else {
            self.node_at(node).left
        }
    }

    #[inline]
    const fn get_right(&self, node: usize) -> usize {
        if node == self.nil {
            self.nil
        } else {
            self.node_at(node).right
        }
    }

    #[cfg(debug_assertions)]
    fn verify_rb_invariants(&self) -> bool {
        if self.root == self.nil {
            return true;
        }

        if !self.is_black(self.root) {
            return false;
        }

        self.verify_black_height(self.root).is_some()
    }

    #[cfg(debug_assertions)]
    fn verify_black_height(&self, node: usize) -> Option<usize> {
        if node == self.nil {
            return Some(1);
        }

        let node_ref = self.node_at(node);

        if self.is_red(node) {
            if !self.is_black(node_ref.left) || !self.is_black(node_ref.right) {
                return None;
            }
        }

        let left_height = self.verify_black_height(node_ref.left)?;
        let right_height = self.verify_black_height(node_ref.right)?;

        if left_height != right_height {
            return None;
        }

        if self.is_black(node) {
            Some(left_height + 1)
        } else {
            Some(left_height)
        }
    }

    pub fn median_absolute_deviation(&self) -> Option<T> {
        if self.total_count == 0 {
            return None;
        }

        let median = self.median()?;

        let mut deviations = Vec::with_capacity(self.total_count);
        self.collect_deviations_to_vec(self.root, median, &mut deviations);

        deviations.sort_unstable_by(|&a, b| a.partial_cmp(b).unwrap());
        let mid = deviations.len() / 2;
        Some(deviations[mid])
    }

    fn collect_deviations_to_vec(&self, node_idx: usize, median: T, deviations: &mut Vec<T>) {
        if node_idx == self.nil {
            return;
        }

        let node = self.node_at(node_idx);
        let value = node.value.into_inner();
        let deviation = (value - median).abs();

        for _ in 0..node.count {
            deviations.push(deviation);
        }

        self.collect_deviations_to_vec(node.left, median, deviations);
        self.collect_deviations_to_vec(node.right, median, deviations);
    }

    pub fn mean_absolute_deviation(&self, mean: T) -> Option<T> {
        if self.total_count == 0 {
            return None;
        }
        let total_deviation = self.sum_absolute_deviations_from_mean(self.root, mean);
        let count_as_t = T::from(self.total_count)?;
        Some(total_deviation / count_as_t)
    }

    /// Sum absolute deviations from mean - O(n)
    fn sum_absolute_deviations_from_mean(&self, node_idx: usize, mean: T) -> T {
        if node_idx == self.nil {
            return T::zero();
        }

        let node = self.node_at(node_idx);
        let value = node.value.into_inner();
        let deviation = (value - mean).abs();

        let count_as_t = match T::from(node.count) {
            Some(count) => count,
            None => return T::zero(),
        };

        let node_deviation = deviation * count_as_t;
        let left_deviation = self.sum_absolute_deviations_from_mean(node.left, mean);
        let right_deviation = self.sum_absolute_deviations_from_mean(node.right, mean);

        node_deviation + left_deviation + right_deviation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rbtree_creation() {
        let tree = RbTree::<f64>::new(10);
        assert_eq!(tree.len(), 0);
        assert_eq!(tree.total_count(), 0);
        assert_eq!(tree.capacity(), 10);
        assert_eq!(tree.remaining_capacity(), 10);
        assert!(tree.is_empty());
        assert!(tree.min().is_none());
        assert!(tree.max().is_none());
        assert!(tree.median().is_none());
    }

    #[test]
    #[should_panic(expected = "Capacity must be greater than 0")]
    fn test_rbtree_zero_capacity() {
        RbTree::<f64>::new(0);
    }

    #[test]
    fn test_rbtree_single_element() {
        let mut tree = RbTree::<f64>::new(5);

        assert!(tree.insert(42.0).is_some());
        assert_eq!(tree.len(), 1);
        assert_eq!(tree.total_count(), 1);
        assert_eq!(tree.remaining_capacity(), 4);
        assert!(!tree.is_empty());

        assert_eq!(tree.min(), Some(42.0));
        assert_eq!(tree.max(), Some(42.0));
        assert_eq!(tree.median(), Some(42.0));
        assert_eq!(tree.quantile(0.0), Some(42.0));
        assert_eq!(tree.quantile(0.5), Some(42.0));
        assert_eq!(tree.quantile(1.0), Some(42.0));

        assert_eq!(tree.remove(42.0), Some(42.0));
        assert_eq!(tree.len(), 0);
        assert_eq!(tree.total_count(), 0);
        assert!(tree.is_empty());
        assert!(tree.min().is_none());
        assert!(tree.max().is_none());
    }

    #[test]
    fn test_rbtree_basic_insert_remove() {
        let mut tree = RbTree::<f64>::new(5);

        assert!(tree.insert(5.0).is_some());
        assert!(tree.insert(3.0).is_some());
        assert!(tree.insert(7.0).is_some());
        assert!(tree.insert(1.0).is_some());
        assert!(tree.insert(9.0).is_some());

        assert_eq!(tree.len(), 5);
        assert_eq!(tree.total_count(), 5);
        assert_eq!(tree.remaining_capacity(), 0);

        assert_eq!(tree.min(), Some(1.0));
        assert_eq!(tree.max(), Some(9.0));

        assert_eq!(tree.remove(3.0), Some(3.0));
        assert_eq!(tree.len(), 4);
        assert_eq!(tree.remove(1.0), Some(1.0));
        assert_eq!(tree.len(), 3);

        assert_eq!(tree.min(), Some(5.0));
        assert_eq!(tree.max(), Some(9.0));

        assert_eq!(tree.remove(100.0), None);
        assert_eq!(tree.len(), 3);
    }

    #[test]
    fn test_rbtree_duplicates() {
        let mut tree = RbTree::<f64>::new(5);

        assert!(tree.insert(5.0).is_some());
        assert!(tree.insert(5.0).is_some());
        assert!(tree.insert(5.0).is_some());

        assert_eq!(tree.len(), 1);
        assert_eq!(tree.total_count(), 3);

        assert_eq!(tree.remove(5.0), Some(5.0));
        assert_eq!(tree.len(), 1);
        assert_eq!(tree.total_count(), 2);

        assert_eq!(tree.remove(5.0), Some(5.0));
        assert_eq!(tree.len(), 1);
        assert_eq!(tree.total_count(), 1);

        assert_eq!(tree.remove(5.0), Some(5.0));
        assert_eq!(tree.len(), 0);
        assert_eq!(tree.total_count(), 0);
        assert!(tree.is_empty());
    }

    #[test]
    fn test_rbtree_capacity_limits() {
        let mut tree = RbTree::<f64>::new(3);

        assert!(tree.insert(1.0).is_some());
        assert!(tree.insert(2.0).is_some());
        assert!(tree.insert(3.0).is_some());
        assert_eq!(tree.remaining_capacity(), 0);

        assert!(tree.insert(4.0).is_none());
        assert_eq!(tree.len(), 3);

        assert!(tree.insert(2.0).is_some());
        assert_eq!(tree.len(), 3);
        assert_eq!(tree.total_count(), 4);
    }

    #[test]
    fn test_rbtree_quantiles_basic() {
        let mut tree = RbTree::<f64>::new(10);

        for i in 1..=5 {
            tree.insert(i as f64).unwrap();
        }

        assert_eq!(tree.quantile(0.0), Some(1.0));
        assert_eq!(tree.quantile(1.0), Some(5.0));
        assert_eq!(tree.quantile(0.25), Some(2.0));
        assert_eq!(tree.quantile(0.5), Some(3.0));
        assert_eq!(tree.quantile(0.75), Some(4.0));
        assert_eq!(tree.median(), Some(3.0));
        assert_eq!(tree.percentile(50.0), Some(3.0));
    }

    #[test]
    fn test_rbtree_quantiles_with_duplicates() {
        let mut tree = RbTree::<f64>::new(10);

        tree.insert(1.0).unwrap();
        tree.insert(2.0).unwrap();
        tree.insert(2.0).unwrap();
        tree.insert(2.0).unwrap();
        tree.insert(5.0).unwrap();

        assert_eq!(tree.total_count(), 5);
        assert_eq!(tree.quantile(0.0), Some(1.0));
        assert_eq!(tree.quantile(0.5), Some(2.0));
        assert_eq!(tree.quantile(1.0), Some(5.0));
    }

    #[test]
    fn test_rbtree_quantiles_edge_cases() {
        let mut tree = RbTree::<f64>::new(10);

        assert!(tree.quantile(0.5).is_none());
        assert!(tree.median().is_none());

        tree.insert(42.0).unwrap();
        assert_eq!(tree.quantile(0.0), Some(42.0));
        assert_eq!(tree.quantile(0.5), Some(42.0));
        assert_eq!(tree.quantile(1.0), Some(42.0));
        assert_eq!(tree.quantile(-0.5), Some(42.0));
        assert_eq!(tree.quantile(1.5), Some(42.0));
    }

    #[test]
    fn test_rbtree_quantiles_even_count() {
        let mut tree = RbTree::<f64>::new(10);

        for i in 1..=4 {
            tree.insert(i as f64).unwrap();
        }

        assert_eq!(tree.total_count(), 4);
        assert_eq!(tree.quantile(0.5), Some(2.0));
    }

    #[test]
    fn test_rbtree_memory_reuse() {
        let mut tree = RbTree::<f64>::new(3);

        tree.insert(1.0).unwrap();
        tree.insert(2.0).unwrap();
        tree.insert(3.0).unwrap();
        assert_eq!(tree.remaining_capacity(), 0);

        tree.remove(1.0).unwrap();
        tree.remove(2.0).unwrap();
        tree.remove(3.0).unwrap();
        assert_eq!(tree.remaining_capacity(), 3);

        tree.insert(4.0).unwrap();
        tree.insert(5.0).unwrap();
        tree.insert(6.0).unwrap();
        assert_eq!(tree.len(), 3);
        assert_eq!(tree.remaining_capacity(), 0);
    }

    #[test]
    fn test_rbtree_sliding_window_basic() {
        let mut tree = RbTree::<f64>::new(3);

        tree.insert(1.0).unwrap();
        tree.insert(2.0).unwrap();
        tree.insert(3.0).unwrap();

        tree.remove(1.0).unwrap();
        tree.insert(4.0).unwrap();

        assert_eq!(tree.len(), 3);
        assert_eq!(tree.min(), Some(2.0));
        assert_eq!(tree.max(), Some(4.0));
    }

    #[test]
    fn test_rbtree_sliding_window_stress() {
        let mut tree = RbTree::<f64>::new(5);
        let inputs = [10.0, 10.5, 11.2, 10.9, 11.5, 11.9, 12.3, 12.1, 11.8, 12.5];

        for (i, &value) in inputs.iter().enumerate() {
            if i >= 5 {
                let removed = tree.remove(inputs[i - 5]);
                assert!(
                    removed.is_some(),
                    "Failed to remove {} at step {}",
                    inputs[i - 5],
                    i
                );
            }

            let inserted = tree.insert(value);
            assert!(
                inserted.is_some(),
                "Failed to insert {} at step {}",
                value,
                i
            );

            assert!(tree.len() <= 5, "Tree exceeded capacity at step {}", i);

            assert!(tree.min().is_some());
            assert!(tree.max().is_some());
            if tree.total_count() > 0 {
                assert!(tree.median().is_some());
            }
        }
    }

    #[test]
    fn test_rbtree_floating_point_precision() {
        let mut tree = RbTree::<f64>::new(10);

        let val1 = 0.1 + 0.2;
        let val2 = 0.3;

        tree.insert(val1).unwrap();
        tree.insert(val2).unwrap();

        assert_eq!(tree.len(), 2);

        assert_eq!(tree.remove(val1), Some(val1));
        assert_eq!(tree.len(), 1);
    }

    #[test]
    fn test_rbtree_infinity_values() {
        let mut tree = RbTree::<f64>::new(10);

        tree.insert(1.0).unwrap();
        tree.insert(f64::INFINITY).unwrap();
        tree.insert(f64::NEG_INFINITY).unwrap();
        tree.insert(2.0).unwrap();

        assert_eq!(tree.len(), 4);
        assert_eq!(tree.min(), Some(f64::NEG_INFINITY));
        assert_eq!(tree.max(), Some(f64::INFINITY));
    }

    #[test]
    fn test_rbtree_large_dataset() {
        let mut tree = RbTree::<f64>::new(1000);

        for i in 0..500 {
            tree.insert(i as f64).unwrap();
        }

        assert_eq!(tree.len(), 500);
        assert_eq!(tree.total_count(), 500);
        assert_eq!(tree.min(), Some(0.0));
        assert_eq!(tree.max(), Some(499.0));
        assert_eq!(tree.median(), Some(249.0));

        for i in (0..500).step_by(2) {
            tree.remove(i as f64).unwrap();
        }

        assert_eq!(tree.len(), 250);
        assert_eq!(tree.total_count(), 250);
    }

    #[test]
    fn test_rbtree_reset_functionality() {
        let mut tree = RbTree::<f64>::new(10);

        for i in 1..=5 {
            tree.insert(i as f64).unwrap();
        }
        assert_eq!(tree.len(), 5);

        tree.reset();
        assert_eq!(tree.len(), 0);
        assert_eq!(tree.total_count(), 0);
        assert_eq!(tree.remaining_capacity(), 10);
        assert!(tree.is_empty());

        tree.insert(42.0).unwrap();
        assert_eq!(tree.len(), 1);
    }

    #[test]
    fn test_rbtree_complex_removal_patterns() {
        let mut tree = RbTree::<f64>::new(10);

        let values = [5.0, 2.0, 8.0, 1.0, 3.0, 7.0, 9.0];
        for &val in &values {
            tree.insert(val).unwrap();
        }

        tree.remove(1.0).unwrap();
        tree.remove(5.0).unwrap();
        tree.remove(8.0).unwrap();

        assert_eq!(tree.len(), 4);
        assert!(tree.min().is_some());
        assert!(tree.max().is_some());
        assert!(tree.median().is_some());
    }

    #[test]
    fn test_rbtree_removal_edge_cases() {
        let mut tree = RbTree::<f64>::new(10);

        tree.insert(5.0).unwrap();

        tree.remove(5.0).unwrap();
        assert!(tree.is_empty());

        for i in 1..=7 {
            tree.insert(i as f64).unwrap();
        }

        tree.remove(4.0).unwrap();
        assert_eq!(tree.len(), 6);

        tree.remove(1.0).unwrap();
        assert_eq!(tree.len(), 5);

        tree.remove(7.0).unwrap();
        assert_eq!(tree.len(), 4);
    }

    #[test]
    fn test_rbtree_error_conditions() {
        let mut tree = RbTree::<f64>::new(2);

        assert_eq!(tree.remove(42.0), None);

        tree.insert(1.0).unwrap();
        tree.insert(2.0).unwrap();

        assert!(tree.insert(3.0).is_none());

        assert!(tree.insert(1.0).is_some());
    }

    #[test]
    fn test_rbtree_sequential_insert_remove() {
        let mut tree = RbTree::<f64>::new(100);

        for i in 0..50 {
            tree.insert(i as f64).unwrap();
        }

        for i in 0..25 {
            tree.remove(i as f64).unwrap();
        }

        assert_eq!(tree.len(), 25);
        assert_eq!(tree.min(), Some(25.0));
        assert_eq!(tree.max(), Some(49.0));
    }

    #[test]
    fn test_rbtree_reverse_order_insertion() {
        let mut tree = RbTree::<f64>::new(10);

        for i in (1..=5).rev() {
            tree.insert(i as f64).unwrap();
        }

        assert_eq!(tree.len(), 5);
        assert_eq!(tree.min(), Some(1.0));
        assert_eq!(tree.max(), Some(5.0));
        assert_eq!(tree.median(), Some(3.0));
    }

    #[test]
    fn test_rbtree_alternating_insert_remove() {
        let mut tree = RbTree::<f64>::new(10);

        tree.insert(5.0).unwrap();
        tree.insert(3.0).unwrap();
        tree.remove(5.0).unwrap();
        tree.insert(7.0).unwrap();
        tree.remove(3.0).unwrap();
        tree.insert(1.0).unwrap();

        assert_eq!(tree.len(), 2);
        assert!(tree.min().is_some());
        assert!(tree.max().is_some());
    }

    #[test]
    fn test_rbtree_duplicate_heavy_workload() {
        let mut tree = RbTree::<f64>::new(5);

        for _ in 0..20 {
            tree.insert(1.0).unwrap();
        }

        assert_eq!(tree.len(), 1);
        assert_eq!(tree.total_count(), 20);

        for _ in 0..15 {
            tree.remove(1.0).unwrap();
        }

        assert_eq!(tree.len(), 1);
        assert_eq!(tree.total_count(), 5);
    }

    #[test]
    fn test_rbtree_find_kth_edge_cases() {
        let mut tree = RbTree::<f64>::new(5);

        tree.insert(1.0).unwrap();
        tree.insert(2.0).unwrap();
        tree.insert(3.0).unwrap();

        assert_eq!(tree.quantile(0.0), Some(1.0));
        assert_eq!(tree.quantile(1.0), Some(3.0));

        assert!(tree.find_kth_element(10).is_none());
    }

    #[test]
    fn test_rbtree_quantile_precision() {
        let mut tree = RbTree::<f64>::new(10);

        for i in 0..10 {
            tree.insert(i as f64).unwrap();
        }

        assert_eq!(tree.quantile(0.1), Some(0.0));
        assert_eq!(tree.quantile(0.9), Some(8.0));
        assert_eq!(tree.quantile(0.55), Some(4.0));
    }

    #[test]
    fn test_rbtree_tree_balance_after_operations() {
        let mut tree = RbTree::<f64>::new(100);

        for i in 0..50 {
            tree.insert(i as f64).unwrap();
        }

        for i in (0..50).step_by(3) {
            tree.remove(i as f64).unwrap();
        }

        for i in 50..75 {
            tree.insert(i as f64).unwrap();
        }

        assert!(tree.min().is_some());
        assert!(tree.max().is_some());
        assert!(tree.median().is_some());
    }

    #[test]
    fn test_rbtree_extreme_values() {
        let mut tree = RbTree::<f64>::new(10);

        tree.insert(f64::MIN).unwrap();
        tree.insert(f64::MAX).unwrap();
        tree.insert(0.0).unwrap();
        tree.insert(-0.0).unwrap();

        assert_eq!(tree.min(), Some(f64::MIN));
        assert_eq!(tree.max(), Some(f64::MAX));
    }

    #[test]
    fn test_rbtree_capacity_one_debug() {
        let mut tree = RbTree::<f64>::new(1);

        assert!(tree.insert(42.0).is_some());
        assert_eq!(tree.len(), 1);
        assert_eq!(tree.total_count(), 1);

        assert!(tree.insert(42.0).is_some());
        assert_eq!(tree.len(), 1);
        assert_eq!(tree.total_count(), 2);

        let node_idx = tree.find_node(42.0).unwrap();
        assert_eq!(tree.node_at(node_idx).count, 2);

        tree.remove(42.0).unwrap();
        assert_eq!(tree.total_count(), 1);

        if let Some(node_idx) = tree.find_node(42.0) {
            assert_eq!(tree.node_at(node_idx).count, 1);
            assert_eq!(tree.len(), 1);
        } else {
            panic!("Node should still exist after first removal");
        }

        tree.remove(42.0).unwrap();
        assert_eq!(tree.len(), 0);
        assert_eq!(tree.total_count(), 0);
        assert!(tree.is_empty());
    }

    #[test]
    fn test_rbtree_capacity_one() {
        let mut tree = RbTree::<f64>::new(1);

        assert!(tree.insert(42.0).is_some());
        assert_eq!(tree.len(), 1);
        assert_eq!(tree.remaining_capacity(), 0);

        assert!(tree.insert(43.0).is_none());
        assert!(tree.insert(42.0).is_some());
        assert_eq!(tree.total_count(), 2);

        tree.remove(42.0).unwrap();
        assert_eq!(tree.total_count(), 1);
        assert_eq!(tree.len(), 1);

        tree.remove(42.0).unwrap();
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
    }

    #[test]
    fn test_rbtree_mixed_duplicate_operations() {
        let mut tree = RbTree::<f64>::new(3);

        tree.insert(1.0).unwrap();
        tree.insert(1.0).unwrap();
        tree.insert(2.0).unwrap();
        tree.insert(2.0).unwrap();
        tree.insert(3.0).unwrap();

        assert_eq!(tree.len(), 3);
        assert_eq!(tree.total_count(), 5);

        tree.remove(1.0).unwrap();
        assert_eq!(tree.total_count(), 4);

        tree.remove(2.0).unwrap();
        tree.remove(2.0).unwrap();
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.total_count(), 2);
    }

    #[test]
    fn test_large_capacity_limits() {
        assert!(std::panic::catch_unwind(|| RbTree::<f64>::new(usize::MAX)).is_err());
    }

    #[test]
    fn test_quantile_precision_large_datasets() {
        let mut tree = RbTree::<f64>::new(1000);

        for i in 0..1000 {
            tree.insert(i as f64).unwrap();
        }

        assert_eq!(tree.quantile(0.5), Some(499.0));
        assert_eq!(tree.quantile(0.999), Some(998.0));
        assert_eq!(tree.quantile(0.001), Some(0.0));
    }

    #[test]
    fn test_deletion_stress() {
        let mut tree = RbTree::<f64>::new(1000);

        for i in 0..500 {
            tree.insert(i as f64).unwrap();
        }

        for i in (0..500).step_by(2) {
            tree.remove(i as f64).unwrap();
        }

        assert_eq!(tree.len(), 250);

        for i in (1..500).step_by(2) {
            assert!(tree.find_node(i as f64).is_some());
        }
    }

    #[test]
    fn test_nan_handling() {
        let mut tree = RbTree::<f64>::new(10);

        tree.insert(1.0).unwrap();
        tree.insert(f64::NAN).unwrap();
        tree.insert(2.0).unwrap();

        assert_eq!(tree.len(), 3);
        assert!(tree.min().is_some());
        assert!(tree.max().is_some());
    }

    #[test]
    fn test_overflow_protection() {
        let mut tree = RbTree::<f64>::new(10);

        tree.insert(1.0).unwrap();

        let node_idx = tree.find_node(1.0).unwrap();
        tree.node_at_mut(node_idx).count = u32::MAX;
        tree.recalculate_subtree_count(node_idx);

        assert!(tree.node_at(node_idx).subtree_count >= u32::MAX as usize);
    }

    #[test]
    fn test_complex_deletion_scenarios() {
        let mut tree = RbTree::<f64>::new(50);

        let values = vec![
            50.0, 25.0, 75.0, 12.0, 37.0, 62.0, 87.0, 6.0, 18.0, 31.0, 43.0,
        ];

        for &val in &values {
            tree.insert(val).unwrap();
        }

        tree.remove(25.0).unwrap();
        tree.remove(50.0).unwrap();
        tree.remove(75.0).unwrap();

        assert_eq!(tree.len(), 8);
        assert!(tree.min().is_some());
        assert!(tree.max().is_some());
    }

    #[test]
    fn test_rb_tree_invariants_stress() {
        let mut tree = RbTree::<f64>::new(100);

        let operations = vec![
            (true, 50.0),
            (true, 25.0),
            (true, 75.0),
            (true, 12.0),
            (true, 37.0),
            (false, 25.0),
            (true, 100.0),
            (false, 50.0),
            (true, 1.0),
            (false, 12.0),
            (true, 200.0),
            (true, 150.0),
            (false, 75.0),
            (true, 300.0),
            (false, 1.0),
        ];

        for (is_insert, value) in operations {
            if is_insert {
                tree.insert(value);
            } else {
                tree.remove(value);
            }
        }

        assert!(tree.len() <= tree.capacity());
        if !tree.is_empty() {
            assert!(tree.min().is_some());
            assert!(tree.max().is_some());
        }
    }

    #[test]
    fn test_subtree_count_consistency() {
        let mut tree = RbTree::<f64>::new(20);

        for i in 0..10 {
            tree.insert(i as f64).unwrap();
            tree.insert(i as f64).unwrap();
        }

        assert_eq!(tree.len(), 10);
        assert_eq!(tree.total_count(), 20);

        for i in 0..5 {
            tree.remove(i as f64).unwrap();
        }

        assert_eq!(tree.len(), 10);
        assert_eq!(tree.total_count(), 15);

        for i in 0..5 {
            tree.remove(i as f64).unwrap();
        }

        assert_eq!(tree.len(), 5);
        assert_eq!(tree.total_count(), 10);
    }

    #[test]
    fn test_edge_case_rotations() {
        let mut tree = RbTree::<f64>::new(10);

        for i in 1..=7 {
            tree.insert(i as f64).unwrap();
        }

        assert_eq!(tree.len(), 7);
        assert_eq!(tree.min(), Some(1.0));
        assert_eq!(tree.max(), Some(7.0));
        assert_eq!(tree.median(), Some(4.0));

        tree.reset();
        for i in (1..=7).rev() {
            tree.insert(i as f64).unwrap();
        }

        assert_eq!(tree.len(), 7);
        assert_eq!(tree.min(), Some(1.0));
        assert_eq!(tree.max(), Some(7.0));
        assert_eq!(tree.median(), Some(4.0));
    }

    #[test]
    fn test_quantile_boundary_conditions() {
        let mut tree = RbTree::<f64>::new(100);

        for i in 0..100 {
            tree.insert(i as f64).unwrap();
        }

        assert_eq!(tree.quantile(0.0), Some(0.0));
        assert_eq!(tree.quantile(1.0), Some(99.0));
        assert_eq!(tree.quantile(0.01), Some(0.0));
        assert_eq!(tree.quantile(0.99), Some(98.0));
        assert_eq!(tree.quantile(0.5), Some(49.0));

        assert_eq!(tree.quantile(0.123), Some(12.0));
        assert_eq!(tree.quantile(0.876), Some(86.0));
    }

    #[test]
    fn test_concurrent_modifications() {
        let mut tree = RbTree::<f64>::new(10);
        tree.insert(1.0).unwrap();
        tree.insert(2.0).unwrap();
        tree.reset();
        tree.insert(3.0).unwrap();
    }

    #[test]
    fn test_quantile_and_percentile_edge_values() {
        let mut tree = RbTree::<f64>::new(10);
        tree.insert(1.0).unwrap();
        tree.insert(2.0).unwrap();
        tree.insert(3.0).unwrap();

        assert!(tree.quantile(f64::NAN).is_some());
        assert!(tree.quantile(f64::INFINITY).is_some());
        assert!(tree.quantile(-5.0).is_some());
        assert!(tree.quantile(5.0).is_some());

        assert!(tree.percentile(f64::NAN).is_some());
        assert!(tree.percentile(-50.0).is_some());
        assert!(tree.percentile(150.0).is_some());
    }

    #[test]
    fn test_many_median() {
        let inputs = [
            0.0, 3.0, 3.0, 1.0, 0.0, 6.0, 6.0, 0.0, 1.0, 2.0, 4.0, 0.0, 2.0, 0.0, 4.0,
        ];
        let mut tree = RbTree::new(3);
        for (idx, &i) in inputs.iter().enumerate() {
            tree.insert(i);
            if idx % 3 == 0 {
                println!("{:?}", tree.median());
            }
        }
    }
}
