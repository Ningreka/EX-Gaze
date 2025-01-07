from typing import Tuple, List, Optional, Union

from mmengine import ConfigDict, Config

class DictPath:

    def __init__(self, key: str = None, children: Optional[List] = None, merge_children=True, unfold_value=False):
        self.key = key # None imply the current node
        self.children: List[DictPath] = children
        self.merge_children = merge_children
        self.unfold_value = unfold_value

    def is_end(self):
        return self.children is None

    def get_dict_value(self, source_dict):
        """
        iter to leaf
        :param source_dict:
        :return:
        """
        if self.key is None:
            cur_value = source_dict
        else:
            cur_value = source_dict[self.key]
        if self.is_end():
            return cur_value
        else:
            children_values = []
            for c in self.children:
                child_value = c.get_dict_value(cur_value)
                if self.merge_children:
                    children_values = [*children_values, *child_value]
                else:
                    children_values.append(child_value)

            return children_values

    def set_dict_value(self, target_dict, target_value):

        if self.is_end():
            if self.key is None:
                raise ValueError("DictPAth.key should not be None when set dict value")
            if self.unfold_value:
                assert len(target_value) == 1, "can only unfold one item value"
                target_dict[self.key] = target_value[0]
            else:
                target_dict[self.key] = target_value
        else:
            cur_dict = target_dict if self.key is None else target_dict[self.key]
            for c in self.children:
                c.set_dict_value(cur_dict, target_value)


def build_dict_path(cfg: Union[dict, ConfigDict, Config]):
    if not isinstance(cfg, (dict, ConfigDict, Config)):
        raise TypeError(
            f'cfg should be a dict, ConfigDict or Config, but got {type(cfg)}')
    if not DictPath == cfg["type"]:
        raise TypeError(
            f'the type in cfg should be DictPath, but get {cfg["type"]}')
    key = cfg.get("key")
    children_cfg = cfg.get("children")
    merge_children = cfg.setdefault("merge_children", True)
    unfold_value = cfg.setdefault("unfold_value", False)
    if children_cfg is not None:
        children = []
        for child_cfg in children_cfg:
            children.append(build_dict_path(child_cfg))
    else:
        children = None
    return DictPath(key=key, children=children, merge_children=merge_children, unfold_value=unfold_value)


def parse_item_in_results(results, key):
    """
    get item by key
    e.g. ("parent","child1","grandson",1)
    :param results:
    :param key:
    :return:
    """
    if isinstance(key, str):
        return results[key]
    elif isinstance(key, (Tuple, List)) and len(key) > 0:
        item = results[key[0]]
        for k in key[1:]:
            item = item[k]
        return item
    else:
        return None

    # end_points = [DictPath(0), DictPath(1)]
    # instances_mask_point = DictPath("instances_mask", end_points,merge_children=False)
    # pre_gt_point = DictPath('pre_gt', instances_mask_point,merge_children=True)
