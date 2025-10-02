import numpy as np
from numpy import linalg as la
from tabulate import tabulate
import pandas as pd
from scipy.stats import t as tdist


def estimate( 
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

    
def est_ols( y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Estimates y on x by ordinary least squares, returns coefficents

    Args:
        >> y (np.ndarray): Dependent variable (Needs to have shape 2D shape)
        >> x (np.ndarray): Independent variable (Needs to have shape 2D shape)

    Returns:
        np.array: Estimated beta coefficients.
    """
    return la.inv(x.T@x)@(x.T@y)

def variance( 
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

def robust( x: np.ndarray, residual: np.ndarray, T:int) -> tuple:
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

def print_table(
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


def perm( Q_T: np.ndarray, A: np.ndarray) -> np.ndarray:
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


def load_example_data():
    # Load the data from firms.csv using pandas
    df = pd.read_csv('firms.csv')

    # Extract variables: 'firmid', 'year', 'lcap', 'lemp', 'ldsa'
    id_array = df['firmid'].values
    year = df['year'].values

    # Count how many firms we have and average time periods
    unique_id = np.unique(id_array, return_counts=True)
    T = int(unique_id[1].mean())

    # Dependent variable: log output (deflated sales)
    y = df['ldsa'].values.reshape(-1, 1)

    # Independent variables: constant, log labour, log capital
    x = np.column_stack([
        np.ones(y.shape[0]),
        df['lemp'].values,
        df['lcap'].values
    ])

    # Variable names (ordered to match the handout tables)
    label_y = 'Log deflated sales'
    label_x = ['Constant', 'Log labour', 'Log capital']

    return y, x, T, year, label_y, label_x


def fd_exogeneity_lead_test(y, x, N, T, cap_col=1, emp_col=2, logs=False, drop_zeros=True):
    """
    FD lead-variable exogeneity test:
        Δy_it = b1 ΔK_it + b2 ΔL_it + b3 ΔL_{i,t+1} + Δu_it

    If logs=True:
        Δlog y_it = b1 Δlog K_it + b2 Δlog L_it + b3 Δlog L_{i,t+1} + Δu_it

    Parameters
    ----------
    y : array, (N*T, 1)
    x : array, (N*T, K)
    N : int, number of panels (firms)
    T : int, number of periods
    cap_col : int, column index for capital
    emp_col : int, column index for employment
    logs : bool, if True takes logs before differencing
    drop_zeros : bool, if True drops firms with non-positive values before log transform

    Prints
    ------
    Coefficients, SEs, t-stats, and decision on H0: b3 = 0.
    """

    K = x.shape[1]
    x3 = x.reshape(N, T, K)
    y2 = y.reshape(N, T, 1)

    # Handle logs
    if logs:
        if drop_zeros:
            # Drop firms with any non-positive values
            valid_firms = np.all(x3[:,:,cap_col] > 0, axis=1) \
                        & np.all(x3[:,:,emp_col] > 0, axis=1) \
                        & np.all(y2[:,:,0] > 0, axis=1)
            x3 = x3[valid_firms]
            y2 = y2[valid_firms]
            N = x3.shape[0]
        x3 = np.log(x3)
        y2 = np.log(y2)

    # First differences
    dx = np.diff(x3, axis=1)
    dy = np.diff(y2, axis=1)

    dcap, demp = dx[:, :, cap_col], dx[:, :, emp_col]

    # Align with lead
    demp_lead = demp[:, 1:]
    demp_cur  = demp[:, :-1]
    dcap_cur  = dcap[:, :-1]
    dy_cur    = dy[:, :-1, 0]

    # Stack
    Y = dy_cur.reshape(-1, 1)
    X = np.c_[np.ones(Y.shape[0]), dcap_cur.ravel(), demp_cur.ravel(), demp_lead.ravel()]

    # OLS
    XtX = X.T @ X
    beta = np.linalg.solve(XtX, X.T @ Y)
    u = Y - X @ beta

    # Cluster-robust SEs (by firm)
    P = X.shape[1]
    meat = np.zeros((P, P))
    rows_per_i = demp_cur.shape[1]
    for i in range(N):
        sl = slice(i*rows_per_i, (i+1)*rows_per_i)
        Xi, ui = X[sl, :], u[sl, :]
        meat += Xi.T @ (ui @ ui.T) @ Xi
    V = np.linalg.inv(XtX) @ meat @ np.linalg.inv(XtX)
    se = np.sqrt(np.diag(V)).reshape(-1,1)
    tvals = (beta / se).flatten()

    # Names
    names = ["const", "ΔCapital", "ΔEmployment", "Lead ΔEmployment"]
    if logs:
        names = ["const", "Δlog Capital", "Δlog Employment", "Lead Δlog Employment"]

    # Print results
    print("FD Exogeneity Test (lead-variable)")
    if logs:
        print("Model: Δ log y_it = b1 Δ log K_it + b2 Δ log L_it + b3 Δ log L_{i,t+1} + Δu_it\n")
    else:
        print("Model: Δ y_it = b1 Δ K_it + b2 Δ L_it + b3 Δ L_{i,t+1} + Δu_it\n")

    print("{:>22}  {:>10}  {:>10}  {:>8}".format("Variable","Beta","SE","t"))
    for nm, b, s, t in zip(names, beta.flatten(), se.flatten(), tvals):
        print(f"{nm:>22}  {float(b):10.4f}  {float(s):10.4f}  {float(t):8.2f}")

    # Test lead coefficient
    b3, se3, t3 = float(beta[-1,0]), float(se[-1,0]), float(tvals[-1])
    df = max(N - 1, 1)
    p3 = float(2*(1 - tdist.cdf(abs(t3), df)))
    print(f"\nTest H0: b3 (Lead term) = 0 → t={t3:.2f}, p={p3:.4g} (df clustered={df})")
    if p3 < 0.05:
        print("→ Reject exogeneity in FD (lead significant).")
    else:
        print("→ Do NOT reject exogeneity in FD.")
