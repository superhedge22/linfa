use ndarray::{s, Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Axis, Data, Ix2};
use ndarray_stats::QuantileExt;
use num_traits::{FromPrimitive, Zero};
use std::collections::HashMap;
use crate::error::{NaiveBayesError, Result};
use linfa::dataset::{AsTargets, DatasetBase, Labels};
use linfa::traits::FitWith;
use linfa::{Float, Label};

// Trait computing predictions for fitted Naive Bayes models
pub(crate) trait NaiveBayes<'a, F, L>
where
    F: Float + PartialOrd,
    L: Label + Ord,
{
    fn joint_log_likelihood(&self, x: ArrayView2<F>) -> HashMap<&L, Array1<F>>;

    fn predict_inplace<D: Data<Elem = F>>(&self, x: &ArrayBase<D, Ix2>, y: &mut Array1<L>)
    where
        F: Float + FromPrimitive + Zero + PartialOrd,
    {
        assert_eq!(
            x.nrows(),
            y.len(),
            "The number of data points must match the number of output targets."
        );
        let joint_log_likelihood = self.joint_log_likelihood(x.view());

        let nclasses = joint_log_likelihood.keys().len();
        let n = x.nrows();
        let mut classes = Vec::with_capacity(nclasses);
        let mut likelihood = Array2::zeros((nclasses, n));

        joint_log_likelihood
            .iter()
            .enumerate()
            .for_each(|(i, (&key, value))| {
                classes.push(key.clone());
                likelihood.row_mut(i).assign(value);
            });

        // Find maximum likelihood for each sample
        *y = (0..n)
            .map(|i| {
                let col = likelihood.column(i);
                let max_idx = (0..nclasses)
                    .max_by(|&i, &j| col[i].partial_cmp(&col[j]).unwrap())
                    .unwrap();
                classes[max_idx].clone()
            })
            .collect::<Array1<_>>();
    }
}

// Common functionality for hyper-parameter sets of Naive Bayes models ready for estimation
pub(crate) trait NaiveBayesValidParams<'a, F, L, D, T>:
FitWith<'a, ArrayBase<D, Ix2>, T, NaiveBayesError>
where
    F: Float + FromPrimitive + Zero + PartialOrd,
    L: Label + Ord,
    D: Data<Elem = F>,
    T: AsTargets<Elem = L> + Labels<Elem = L>,
{
    fn fit(
        &self,
        dataset: &'a DatasetBase<ArrayBase<D, Ix2>, T>,
        model_none: Self::ObjectIn,
    ) -> Result<Self::ObjectOut> {
        let mut unique_classes = dataset.targets.labels();
        unique_classes.sort_unstable();
        self.fit_with(model_none, dataset)
    }
}

pub fn filter<F: Float + PartialOrd, L: Label + Ord>(
    x: ArrayView2<F>,
    y: ArrayView1<L>,
    ycondition: &L,
) -> Array2<F> {
    let index = y
        .into_iter()
        .enumerate()
        .filter_map(|(i, y)| (*ycondition == *y).then(|| i))
        .collect::<Vec<_>>();

    let mut xsubset = Array2::zeros((index.len(), x.ncols()));
    index
        .into_iter()
        .enumerate()
        .for_each(|(i, r)| xsubset.row_mut(i).assign(&x.slice(s![r, ..])));
    xsubset
}