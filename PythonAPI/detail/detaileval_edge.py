import json
import numpy as np
from scipy import ndimage as ndi
from detail import benchmark_ext

lut1 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,1,1,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,1,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,1,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,1,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,0,0,0,0,0,0,0,1,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,0,1,0,0,0,1,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,1,1,0,0,1,1,0,0], dtype=np.bool)

lut2 = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,0,1,0,1,1,0,0,0,0,1,0,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,0,0,1,1,0,0,1,0,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,1,0,0,0,1,1,0,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,0,1,0,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,0,1,0,0,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0,0,1,1,0,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,0,1,0,0,1,0,1,1,0,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,1,0,1,1,1,0,1,0,1,0,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,0,1,0,1,0,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,0,1,0,0,1,0,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,0,1,0,0,1,0,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], dtype=np.bool)

def applylut(x, LUT):
    mask = np.array([[ 256,  32,  4],
                     [128,  16,  2],
                     [64, 8,1]],dtype=np.int)
    corr = ndi.correlate(x.astype(int), mask, mode='constant')
    return np.take(LUT, corr)

def morph(x):
    return np.multiply(x, applylut (applylut (x, lut1), lut2))

def bwmorph(bw):
    while True:
        bw = bw.astype(int)
        premorph = np.sum(bw)
        bw2_tmp = morph(bw)
        postmorph = np.sum(bw2_tmp)
        if premorph == postmorph:
            break
        bw = bw2_tmp;
    return bw

def f1_score(tp, fp, fn):
	if tp == 0:
		return 0
	precision = tp*1.0/(tp + fp)
	recall = tp*1.0/(tp + fn)

	return 2.0/(1/(precision) + 1/(recall))

def match(prediction, ground_truth, oc, max_dist):
	rows, cols = prediction.shape
	pred_match, gt_match, match_cost = benchmark_ext.benchmark(prediction.reshape(rows * cols).tolist(), ground_truth.reshape(rows * cols).tolist(), [cols, rows], [max_dist, oc])

	pred_match = np.array(pred_match).reshape(rows, cols)
	gt_match = np.array(gt_match).reshape(rows, cols)

	tp = np.sum(pred_match > 0)
	fp = np.sum(prediction - (pred_match>0))
	fn = np.sum(ground_truth - (gt_match > 0) * 1)
	return (tp, fp, fn, match_cost)
	

class edgeDetectionEval:
    def __init__(self,details, step = 0.1, max_dist = 0.001, verbose = True):
        self.details = details
        self.json_loaded = False
        self.step = step
        self.max_dist = max_dist
        self.verbose = verbose

    def loadJSON(self, resFile):
        self.data = json.load(open(resFile))
        self.json_loaded = True

    def _get_f1_curve(self, details, data, step, thresh):
        accum_step = step
        results = []
        if self.verbose:
            from progressbar import ProgressBar
            is_first = True
        while accum_step < 1:
            tp_accum = 0
            fn_accum = 0
            fp_accum = 0
            if self.verbose:
                print("Evaluating for threshhold: " + str(accum_step))
                if is_first:
                    pbar = ProgressBar(maxval=len(data)+1).start()
                    is_first = False
                pbar_counter = 0
            for elem in data:
                mask = bwmorph(details.getBounds(str(elem['name']), show=False))
                prediction = bwmorph(1 * (np.array(elem['mask']) > accum_step))
                tp, fp, fn, match_cost = match(prediction, mask, 100, thresh)
                tp_accum = tp_accum + tp
                fp_accum = fp_accum + fp
                fn_accum = fn_accum + fn
                if self.verbose:
                    pbar.update(pbar_counter)
                    pbar_counter += 1
            if self.verbose:
                pbar.finish()
            results.append(f1_score(tp_accum, fp_accum, fn_accum))
            accum_step = accum_step + step
        return results

    def evaluate(self):
        if not self.json_loaded:
            print("You need to load json first with loadJSON")
            exit(1)
        self.result = self._get_f1_curve(self.details, self.data, self.step, self.max_dist)
