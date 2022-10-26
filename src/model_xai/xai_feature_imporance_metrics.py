import numpy as np


class feature_importance_metrics:
    """
    Explainable metrics based on  feature importance score of Explainable AI techniques.
    """

    def faithfulness_metric(self, model, x, coefs, base):

        """
        This metric evaluates the correlation between the importance assigned by the interpretability algorithm
        to attributes and the effect of each of the attributes on the performance of the predictive model.
        The higher the importance, the higher should be the effect, and vice versa, The metric evaluates this by
        incrementally removing each of the attributes deemed important by the interpretability metric, and
        evaluating the effect on the performance, and then calculating the correlation between the weights (importance)
        of the attributes and corresponding model performance. [#]_
        References:
            .. [#] `David Alvarez Melis and Tommi Jaakkola. Towards robust interpretability with self-explaining
               neural networks. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors,
               Advances in Neural Information Processing Systems 31, pages 7775-7784. 2018.
               <https://papers.nips.cc/paper/8003-towards-robust-interpretability-with-self-explaining-neural-networks.pdf>`_
        :param model: Trained classifier, such as a ScikitClassifier that implements
                        a predict() and a predict_proba() methods.
        :param x: (numpy.ndarray) row of data.
        :param coefs: (numpy_ndarray). coefficients (weights) corresponding to feature (attribute) importance
        :param base: (numpy.ndarray). base default values of features(attributes)
        :return: (float) correlation between feature importance weights and corresponding effect on classifiers
        """
        # find predicted class
        pred_class = np.argmax(model.predict_proba(x.reshape(1, -1)), axis=1)[0]

        # find indexes of coefficients in decreasing order of value
        # argsort returns indexes of values sorted in increasing order; so do it for negated array
        ar = np.argsort(-coefs)
        pred_probs = np.zeros(x.shape[0])
        for ind in np.nditer(ar):
            x_copy = x.copy()
            x_copy[ind] = base[ind]
            x_copy_pr = model.predict_proba(x_copy.reshape(1, -1))
            pred_probs[ind] = x_copy_pr[0][pred_class]

        return -np.corrcoef(coefs, pred_probs)[0, 1]

    def monotonicity_metric(self, model, x, coefs, base):
        """
        This metric measures the effect of individual features on model performance by evaluating the effect on
        model performance of incrementally adding each attribute in order of increasing importance. As each feature
        is added, the performance of the model should correspondingly increase, thereby resulting in monotonically
        increasing model performance. [#]_
        References:
            .. [#] `Ronny Luss, Pin-Yu Chen, Amit Dhurandhar, Prasanna Sattigeri, Karthikeyan Shanmugam, and
               Chun-Chen Tu. Generating Contrastive Explanations with Monotonic Attribute Functions. CoRR abs/1905.13565. 2019.
               <https://arxiv.org/pdf/1905.12698.pdf>`_

        :param model: Trained Classifier, such as Scikit-learn classifier that implements a predict() and  predict_prob()
        :param x: (numpy.ndarray) row of data.
        :param coefs:(numpy.ndarray) co-efficients(weights) corresponding to attribute importance.
        :param base: ((numpy.ndarray) base (default) values of features(attributes)
        :return: True if the relationship is monotonic.
        """
        # find predicted class
        pred_class = np.argmax(model.predict_proba(x.reshape(1, -1)), axis=1)[0]

        x_copy = base.copy()

        # find indexes of coefficients in increasing order of value
        ar = np.argsort(coefs)
        pred_probs = np.zeros(x.shape[0])
        for ind in np.nditer(ar):
            x_copy[ind] = x[ind]
            x_copy_pr = model.predict_proba(x_copy.reshape(1, -1))
            pred_probs[ind] = x_copy_pr[0][pred_class]

        return np.all(np.diff(pred_probs[ar]) >= 0)

