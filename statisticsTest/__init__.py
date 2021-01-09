#
#   Statistics test
#

import numpy as np
from scipy.stats.distributions import chi2
import warnings
from functools import reduce
from operator import mul

# chisquare_trend_contingency

def chisquare_trend_contingency (observed, tendencies_values, axis = None):

    """
        chisquare_trend_contingency
        Apply a Chi2 square Cochran-Armitage trend test with variable in a contingency table.
        It measure the effect of an tendencie variable on a evaluated variable.
        The evaluated variable should be a categorial variable with 2 modalities. It should be of shape 2.
        The tendencies values should be of shape >= 2 an be numeric. 

        Validation conditions
        ----------
            - Monotonous evolution of the tendencie (not checked by the function)
            - Theorical value >= 5 : checked by the function

        Parameters
        ----------
            observed, numpy matrix : contingency table of observed value. Should have an axis with a shape of 2 as the test is designed to evaluate tendencies on 2 modalities categorial variable. If only an axis is of shape 2, this axis is used for applying the test, otherwise you should specify the axis containing the tested variable with the axis parameter.
            tendencies_values, numpy vector : values of the tendencie variable, should be an 1D vector of shape >= 2

        Returns
        ----
        chi2 : float
            The test statistic.
        p : float
            The p-value of the test
        dof : int
            Degrees of freedom
        expected : ndarray, same shape as `observed`
            The expected frequencies, based on the marginal sums of the table.
    """

    # Making all numpy
    observed = np.array(observed)
    tendencies_values = np.array(tendencies_values)

    # Checking data

    ## Observed
    observed_shape = np.array(observed.shape)
    if ((observed_shape == 2).sum() == 0):
        # Exception if absence of shape 2
        raise Exception("Observed should contains at list of element of shape 2.")
    
    ## Tendencies values
    if(tendencies_values.ndim != 1):
        # Exception if dim different of 1
        raise Exception("Tendencies values should be an 1D array")
    elif(tendencies_values.shape not in observed_shape):
        raise Exception("Tendencies values should be contains the same number of element than observed")

    # Calculating parameters
    n = observed.sum() # Number of observations
    
    ## Getting axis containing the evaluated variable
    tmp_axis_id = (np.array(observed.shape) == 2) ## Check which axis is of shape 2
    if (tmp_axis_id.sum() == 1):
        # If there is only one axis with shape 2 : it is the evaluated variable
        axis_id = np.where(tmp_axis_id)[0][0]
    else:
        # Taking the content of the axis variables
        axis_id = axis

        if (type(axis_id) != type(int())):
            raise Exception("Observed values are of shape (2,2), you should specify the axis of the evaluated variable with the axis parameter (0 or 1)")

    # Getting theorical values
    expected = (observed.sum(axis = 1)/n).reshape(-1, 1)*observed.sum(axis = 0)

    # Getting dof
    dof = 1 # By desing in this test, this is a 1 dof

    # Checking condition parameter
    ## Every expected value should be >= 5
    if ((expected < 5).sum() != 0):
        warnings.warn("Test condition not respected, every expected value should be >= 5")

    # Getting test parameter
    chi2_parameter = (
        (
            np.power(n, 3)
            *np.square(sum(tendencies_values*(np.take(observed, 0, axis_id)-np.take(expected, 0, axis_id))))
        )/(
            reduce(mul, observed.sum(axis = (1-axis_id)))*
            (
                n*(observed.sum(axis = axis_id)*np.square(tendencies_values)).sum()
                -np.square((observed.sum(axis = axis_id)*(tendencies_values)).sum())
            )
        )
    )

    # Getting p
    p = chi2.sf(chi2_parameter, dof)

    # Returning result
    return(chi2_parameter, p, dof, expected)