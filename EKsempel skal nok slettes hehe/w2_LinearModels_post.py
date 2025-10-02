import numpy as np
from numpy import linalg as la
from tabulate import tabulate



def estimate(y: np.ndarray, x: np.ndarray, transform='', N=None, T=None, robust_se=False) -> list:
    """Takes some np.arrays and estimates regular OLS, FE or FD.
    

    Args:
        y (np.ndarray): The dependent variable, needs to have the shape (n*t, 1)
        x (np.ndarray): The independent variable(s). If only one independent 
        variable, then it needs to have the shape (n*t, 1).
        transform (str, optional): Specify if estimating fe or fd, in order 
        to get correct variance estimation. Defaults to ''.
        N (int, optional): Number of observations. If panel, then the 
        number of individuals. Defaults to None.
        T (int, optional): If panel, then the number of periods an 
        individual is observerd. Defaults to None.

    Returns:
        dict: A dictionary with the results from the ols-estimation.
    """
    
    b_hat = est_ols(y, x)
    resid = y - x@b_hat
    SSR = resid.T@resid
    SST = (y - np.mean(y)).T@(y - np.mean(y))
    R2 = 1 - SSR/SST

    sigma, cov, se = variance(transform, SSR, x, N, T)
    if robust_se:
        cov, se = robust(x, resid, T)
    t_values = b_hat/se
    
    names = ['b_hat', 'se', 'sigma', 't_values', 'R2', 'cov']
    results = [b_hat, se, sigma, t_values, R2, cov]
    return dict(zip(names, results))

    
def est_ols( y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Estimates OLS using input arguments.

    Args:
        y (np.ndarray): Check estimate()
        x (np.ndarray): Check estimate()

    Returns:
        np.array: Estimated beta hats.
    """
    return la.inv(x.T@x)@(x.T@y)
    # alternatively and more efficiently: 
    # return la.solve(x.T@x, x.T@y)

def variance( 
        transform: str, 
        SSR: float, 
        x: np.ndarray, 
        N: int,
        T: int
    ) -> tuple :
    """Use SSR and x array to calculate different variation of the variance.

    Args:
        transform (str): Specifiec if the data is transformed in any way.
        SSR (float): SSR
        x (np.ndarray): Array of independent variables.
        N (int, optional): Number of observations. If panel, then the 
        number of individuals. Defaults to None.
        T (int, optional): If panel, then the number of periods an 
        individual is observerd. Defaults to None.

    Raises:
        Exception: [description]

    Returns:
        tuple: [description]
    """

    # Number of coefficients
    K=x.shape[1]

    if transform in ('', 're', 'fd'):
          sigma = SSR/(N*T-K)
    elif transform.lower() == 'fe':
          sigma = SSR/(N*T-N-K) 
    elif transform.lower() in ('be'): 
          sigma = SSR/(N-K) 
    else:
        raise Exception('Invalid transform provided.')
    
    cov = sigma*la.inv(x.T@x)
    se = np.sqrt(cov.diagonal()).reshape(-1, 1)
    return sigma, cov, se




def robust(x: np.ndarray, residual: np.ndarray, T: int):
    """Panel-robust covariance (clustered by individual)."""
    if (not T) or (T == 1):
        Ainv = la.inv(x.T @ x)
        uhat2 = residual ** 2
        uhat2_x = uhat2 * x
        cov = Ainv @ (x.T @ uhat2_x) @ Ainv
    else:
        nobs, K = x.shape
        N = int(nobs / T) if T else 0
        B = np.zeros((K, K))
        for i in range(N):
            idx_i = slice(i * T, (i + 1) * T)
            Omega = residual[idx_i] @ residual[idx_i].T
            B += x[idx_i].T @ Omega @ x[idx_i]
        Ainv = la.inv(x.T @ x)
        cov = Ainv @ B @ Ainv
    se = np.sqrt(np.diag(cov)).reshape(-1, 1)
    return cov, se

def print_table(labels: tuple, results: dict, headers=None, title="Results", **kwargs) -> None:
    """
    The `print_table` function takes in labels, results, headers, and title as arguments, and prints a
    table with the results and data for model specification.
    
    Args:
      labels (tuple): The `labels` parameter is a tuple that contains the labels for the dependent
    variable and the independent variables. The first element of the tuple is the label for the
    dependent variable, and the remaining elements are the labels for the independent variables.
      results (dict): The `results` parameter is a dictionary that comes from the `estimate` function.
      headers: The `headers` parameter is a list that specifies the column headers for the table. By
    default, it is set to `["", "Beta", "Se", "t-values"]`.
      title: The title of the table, which is "Results" by default. Defaults to Results
    """

    # Unpack labels
    label_y, label_x = labels
    
    # Default headers if not provided
    if headers is None:
        headers = ["", "Beta", "Se", "t-values"]

    # Create table for data on coefficients
    table = []
    for i, name in enumerate(label_x):
        row = [
            name, 
            results.get('b_hat')[i], 
            results.get('se')[i], 
            results.get('t_values')[i]
        ]
        table.append(row)
    
    # Print table
    print(title)
    print(f"Dependent variable: {label_y}\n")
    print(tabulate(table, headers, **kwargs))
    
    # Print data for model specification
    print(f"R² = {results.get('R2').item():.3f}")
    print(f"σ² = {results.get('sigma').item():.3f}")
    
    
def perm( Q_T: np.ndarray, A: np.ndarray, t=0) -> np.ndarray:
    """Takes a transformation matrix and performs the transformation on 
    the given vector or matrix.

    Args:
        Q_T (np.array): The transformation matrix. Needs to have the same
        dimensions as number of years a person is in the sample.
        
        A (np.array): The vector or matrix that is to be transformed. Has
        to be a 2d array.
        
        t (int, optional): The number of years an individual is in the sample.

    Returns:
        np.array: Returns the transformed vector or matrix.
    """
    # We can infer T from the shape of the transformation matrix.
    if t==0:
        t = Q_T.shape[1]

    # Initialize the numpy array
    Z = np.array([[]])
    Z = Z.reshape(0, A.shape[1])

    # Loop over the individuals, and permutate their values.
    for i in range(int(A.shape[0]/t)):
        Z = np.vstack((Z, Q_T@A[i*t: (i + 1)*t]))
    return Z


def estimate1( 
        y: np.ndarray, x: np.ndarray, transform='', T:int=None, robust_se=False
    ) -> list:
    """Uses the provided estimator to perform a regression of y on x, 
    and provides all other necessary statistics such as standard errors, 
    t-values etc.  

    Args:
        >> y (np.ndarray): Dependent variable (Needs to have shape 2D shape)
        >> x (np.ndarray): Independent variable (Needs to have shape 2D shape)
        >> transform (str, optional): Defaults to ''. If the data is 
        transformed in any way, the following transformations are allowed:
            '': No transformations
            'fd': First-difference
            'be': Between transformation
            'fe': Within transformation
            're': Random effects estimation.
        >>T (int, optional): If panel data, T is the number of time periods in
        the panel, and is used for estimating the variance. Defaults to None.
        >> robust_se (bool): Defaults to False. Returns robust standard errors if True.

    Returns:
        list: Returns a dictionary with the following variables:
        'b_hat', 'se', 'sigma2', 't_values', 'R2', 'cov'
    """

    assert y.ndim == 2, 'Input y must be 2-dimensional'
    assert x.ndim == 2, 'Input x must be 2-dimensional'
    assert y.shape[1] == 1, 'y must be a column vector'
    assert y.shape[0] == x.shape[0], 'y and x must have same first dimension'
    
    b_hat = est_ols(y, x)  # Estimated coefficients
    residual = y - x@b_hat  # Calculated residuals
    SSR = residual.T@residual  # Sum of squared residuals
    SST = (y - np.mean(y)).T@(y - np.mean(y))  # Total sum of squares
    R2 = 1 - SSR/SST

    sigma2, cov, se = variance(transform, SSR, x, T)
    # Overwrites cov and se with robust version if specified 'robust_se = True'
    if robust_se:
        cov, se = robust(x, residual, T)
    t_values = b_hat/se
    
    names = ['b_hat', 'se', 'sigma2', 't_values', 'R2', 'cov']
    results = [b_hat, se, sigma2, t_values, R2, cov]
    return dict(zip(names, results))

    
def est_ols1( y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Estimates y on x by ordinary least squares, returns coefficents

    Args:
        >> y (np.ndarray): Dependent variable (Needs to have shape 2D shape)
        >> x (np.ndarray): Independent variable (Needs to have shape 2D shape)

    Returns:
        np.array: Estimated beta coefficients.
    """
    return la.inv(x.T@x)@(x.T@y)

def variance1( 
        transform: str, 
        SSR: float, 
        x: np.ndarray, 
        T: int
    ) -> tuple:
    """Calculates the covariance and standard errors from the OLS
    estimation.

    Args:
        >> transform (str): Defaults to ''. If the data is transformed in 
        any way, the following transformations are allowed:
            '': No transformations
            'fd': First-difference
            'be': Between transformation
            'fe': Within transformation
            're': Random effects estimation
        >> SSR (float): Sum of squared residuals
        >> x (np.ndarray): Dependent variables from regression
        >> t (int): The number of time periods in x.

    Raises:
        Exception: If invalid transformation is provided, returns
        an error.

    Returns:
        tuple: Returns the error variance (mean square error), 
        covariance matrix and standard errors.
    """

    # Store n and k, used for DF adjustments.
    K = x.shape[1]
    if transform in ('', 'fd', 'be'):
        N = x.shape[0]
    else:
        N = x.shape[0]/T

    # Calculate sigma2
    if transform in ('', 'fd', 'be'):
        sigma2 = (np.array(SSR/(N - K)))
    elif transform.lower() == 'fe':
        sigma2 = np.array(SSR/(N * (T - 1) - K))
    elif transform.lower() == 're':
        sigma2 = np.array(SSR/(T * N - K))
    else:
        raise Exception('Invalid transform provided.')
    
    cov = sigma2*la.inv(x.T@x)
    se = np.sqrt(cov.diagonal()).reshape(-1, 1)
    return sigma2, cov, se

def robust1( x: np.ndarray, residual: np.ndarray, T:int) -> tuple:
    '''Calculates the robust variance estimator 

    Args: 
        x: (NT,K) matrix of regressors. Assumes that rows are sorted 
            so that x[:T, :] is regressors for the first individual, 
            and so forth. 
        residual: (NT,1) vector of residuals 
        T: number of time periods. If T==1 or T==None, assumes cross-sectional 
            heteroscedasticity-robust variance estimator
    
    Returns
        tuple: cov, se 
            cov: (K,K) panel-robust covariance matrix 
            se: (K,1) vector of panel-robust standard errors
    '''

    # If only cross sectional, we can use the diagonal.
    if (not T) or (T == 1):
        Ainv = la.inv(x.T@x) 
        uhat2 = residual ** 2
        uhat2_x = uhat2 * x # elementwise multiplication: avoids forming the diagonal matrix (RAM intensive!)
        cov = Ainv @ (x.T@uhat2_x) @ Ainv
    
    # Else we loop over each individual.
    else:
        nobs,K = x.shape
        N = int(nobs / T)
        B = np.zeros((K, K)) # initialize 

        for i in range(N):
            idx_i = slice(i*T, (i+1)*T) # index values for individual i 
            Omega = residual[idx_i]@residual[idx_i].T # (T,T) matrix of outer product of i's residuals 
            B += x[idx_i].T @ Omega @ x[idx_i] # (K,K) contribution 

        Ainv = la.inv(x.T @ x)
        cov = Ainv @ B @ Ainv
    
    se = np.sqrt(np.diag(cov)).reshape(-1, 1)
    return cov, se

def print_table1(
        labels: tuple,
        results: dict,
        headers=["", "Beta", "Se", "t-values"],
        title="Results",
        _lambda:float=None,
        **kwargs
    ) -> None:
    """Prints a nice looking table, must at least have coefficients, 
    standard errors and t-values. The number of coefficients must be the
    same length as the labels.

    Args:
        >> labels (tuple): Touple with first a label for y, and then a list of 
        labels for x.
        >> results (dict): The results from a regression. Needs to be in a 
        dictionary with at least the following keys:
            'b_hat', 'se', 't_values', 'R2', 'sigma2'
        >> headers (list, optional): Column headers. Defaults to 
        ["", "Beta", "Se", "t-values"].
        >> title (str, optional): Table title. Defaults to "Results".
        _lambda (float, optional): Only used with Random effects. 
        Defaults to None.
    """
    
    # Unpack the labels
    label_y, label_x = labels
    
    # Create table, using the label for x to get a variable's coefficient,
    # standard error and t_value.
    table = []
    for i, name in enumerate(label_x):
        row = [
            name, 
            results.get('b_hat')[i], 
            results.get('se')[i], 
            results.get('t_values')[i]
        ]
        table.append(row)
    
    # Print the table
    print(title)
    print(f"Dependent variable: {label_y}\n")
    print(tabulate(table, headers, **kwargs))
    
    # Print extra statistics of the model.
    print(f"R\u00b2 = {results.get('R2').item():.3f}")
    print(f"\u03C3\u00b2 = {results.get('sigma2').item():.3f}")
    if _lambda: 
        print(f'\u03bb = {_lambda.item():.3f}')


def perm1( Q_T: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Takes a transformation matrix and performs the transformation on 
    the given vector or matrix.

    Args:
        Q_T (np.ndarray): The transformation matrix. Needs to have the same
        dimensions as number of years a person is in the sample.
        
        A (np.ndarray): The vector or matrix that is to be transformed. Has
        to be a 2d array.

    Returns:
        np.array: Returns the transformed vector or matrix.
    """
    # We can infer t from the shape of the transformation matrix.
    M,T = Q_T.shape 
    N = int(A.shape[0]/T)
    K = A.shape[1]

    # initialize output 
    Z = np.empty((M*N, K))
    
    for i in range(N): 
        ii_A = slice(i*T, (i+1)*T)
        ii_Z = slice(i*M, (i+1)*M)
        Z[ii_Z, :] = Q_T @ A[ii_A, :]

    return Z

