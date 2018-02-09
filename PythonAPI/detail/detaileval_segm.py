import numpy as np
import json

class baseSegEvalClass(object):
    def __init__(self, details, verbose = True):
        self.details = details
        self.gt_dict = self._json_to_dict(self.details.data)
        self.data_loaded = False
        self.verbose = verbose

    #requires path to file with predictions
    def loadJSON(self, resFile):
        self.data = json.load(open(resFile))
        self.pred_dict = self._json_to_dict(self.data)
        self.cats = self._get_cats(self.pred_dict)
        self.data_loaded = True
        
        
    # Gets list of categories existing in prediction file
    def _get_cats(self, prediction):
        cats = set()
        for elem in prediction.values():
            cats = set.union(cats, set(elem.keys()))
        return list(cats)

    # Takes a JSON file and transforms it into dictionary which makes it easy to search for maps 
    # on certain image for certain class.
    def _json_to_dict(self, data):
        raise NotImplementedError()
    
    # given a 3D tensor (list of maps) returns (x,y) coordinates of non 0 pixels in any of the maps
    def _list_to_indexes(self, list_of_maps):
        total_list = set()
        for elem in list_of_maps:
            temp = self.details.decodeMask(elem)
            x = np.where(temp>0)
            pred_index = set(zip(x[0].tolist(),x[1].tolist()))
            total_list = set.union(total_list, pred_index)
        return total_list

    # Computes intersection and union given dicts returned by _json_to_dict
    def _IU(self, dict1, dict2, cat):
        if (cat in dict1.keys()) and (cat in dict2.keys()):
            set1 = self._list_to_indexes(dict1[cat])
            set2 = self._list_to_indexes(dict2[cat])
            intersection = len(set.intersection(set1, set2))
            union = len(set.union(set1, set2))
        elif cat in dict1.keys():
            set1 = self._list_to_indexes(dict1[cat])
            intersection = 0
            union = len(set1)
        elif cat in dict2.keys():
            set1 = self._list_to_indexes(dict2[cat])
            intersection = 0
            union = len(set1)
        else:
            intersection = 0
            union = 0
        return (intersection, union)
    
    # Performs computation of IoU between ground truth available in details and prediction
    # in file previously loaded with loadJSON. Writes the result under self.results
    def evaluate(self):
        if not self.data_loaded:
            print("Please, load the data with loadJSON first")
            return -1
        cats = self.cats
        dict_pred = self.pred_dict
        dict_gt = self.gt_dict
        intersection = np.zeros(len(cats))
        union = np.zeros(len(cats))
        j = 0
        if self.verbose:
            from progressbar import ProgressBar
            pbar = ProgressBar(maxval=len(dict_pred.keys())).start()
        for img in dict_pred.keys():
            dict1 = dict_pred[img]
            dict2 = dict_gt[img]
            for i in range(len(cats)):
                cat = cats[i]
                temp_int, temp_un = self._IU(dict1, dict2, cat)
                intersection[i] = intersection[i] + temp_int
                union[i] = union[i] + temp_un
            if self.verbose:
                pbar.update(j)
            j = j + 1
        pbar.finish()

        self.intersection = intersection
        self.union = union
        self.results = np.mean(intersection/union)   

        
# Class for evaluation of segmentation. dict[image][category] gives a list of maps (@D arrays).
# Each map is a map of a single instance of the category on the image.
class catSegEvalClass(baseSegEvalClass):
    def _json_to_dict(self, data):
        final_dict = {}
        for elem in data['annos_segmentation']:
            img = elem['image_id']
            if not img in final_dict.keys():
                final_dict[img] = {}
            if elem['parts'] == []:
                cat_id = elem['category_id']
                if cat_id in final_dict[img].keys():
                    final_dict[img][cat_id].append(elem['segmentation'])
                else:
                    final_dict[img][cat_id] = [elem['segmentation']]
        return final_dict

# Class for evaluation of segmentation. dict[image][category or part] gives a list of maps (@D arrays).
# Each map is a map of a single instance of the category or part on the image.
# Here category is available in the dict only if it does not contains of any parts.
class partsSegEvalClass(baseSegEvalClass):
   
    def _map_on_cluster(self, x, clusters):
        for elem in clusters:
            if x in elem:
                return np.min(np.array(elem))
        return x
        
    def _json_to_dict(self, data):
        self.clusters = [[1], [2], [3], [4], [5, 18], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], [19], [20], [21], [22], [23], [24], [25], [47], [48], [49], [50], [51], [52], [26,53], [27,54], [28, 55], [29,56], [30, 57], [31, 58], [32, 67], [33,68], [34, 59], [35, 60], [36, 61], [37, 62], [38,63], [39, 65], [40,66], [41, 69], [42, 70], [43, 71], [72], [44,73], [45, 74], [46, 75], [64], [76], [77], [78], [79], [80], [81], [255]]

        final_dict = {}
        for elem in data['annos_segmentation']:
            img = elem['image_id']
            if not img in final_dict.keys():
                final_dict[img] = {}
            if elem['parts'] == []:
                # Unfortunately cats id overlap with parts id so we need to add some margin to be able to distinguish (300 is enough).
                cat_id = 300 + elem['category_id']
                if cat_id in final_dict[img].keys():
                    final_dict[img][cat_id].append(elem['segmentation'])
                else:
                    final_dict[img][cat_id] = [elem['segmentation']]
            else:
                for part in elem['parts']:
                    # Some parts are redundant so we want to merge them
                    part_id = self._map_on_cluster(part['part_id'], self.clusters)
                    if part_id in final_dict[img].keys():
                        final_dict[img][part_id].append(part['segmentation'])
                    else:
                        final_dict[img][part_id] = [part['segmentation']]
        return final_dict
