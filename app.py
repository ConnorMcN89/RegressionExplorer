# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 13:04:44 2026

@author: conno
"""

#Start by importing the packages that we will use.
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import (
    mean_squared_error, r2_score, confusion_matrix,
    accuracy_score, recall_score, precision_score
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#set page settings
st.set_page_config(page_title="Regression Explorer", layout="centered")
st.title("Regression Explorer")
st.write("Upload a CSV to begin your analysis.")

#Add a file picker button
file = st.file_uploader("Choose a CSV file", type="csv")
if file is None:
    st.info("Upload a dataset.")
    st.stop()
#reads csv file from the uploaded file
data = pd.read_csv(file)
# Convert horsepower to numeric values
if "horsepower" in data.columns:
    data["horsepower"] = pd.to_numeric(data["horsepower"], errors="coerce")
#Add subhead
st.subheader("Data overview")
#Display text
st.write("Dimensions (rows, columns):", data.shape)
#display the first 10 rows in a streamlit dataframe
edited_data = st.data_editor(data, num_rows="dynamic")
st.write("Column types:")
st.dataframe(data.dtypes.astype(str))
st.write("Summary statistics:")
st.dataframe(data.describe(include="all").T)
st.write("Missing values:")
st.dataframe(data.isna().sum())



# Checks for numeric values and returns an error if none are found
numeric = data.select_dtypes(include=np.number).columns.tolist()
if not numeric:
    st.error("No numeric response variable found.")
    st.stop()

# Creates a select box for the response variable and lists candidates
response = st.selectbox("Response variable", numeric)
candidates = [c for c in data.columns if c != response]

# Creates another box for predictors using the candidates
predictors = st.multiselect("Predictor variables", candidates)

# Identifies default categorical variables
default_cats = [c for c in predictors
                if c not in numeric or c == "origin"]

# Creates a dropdown to select which predictors are categorical
categorical = st.multiselect(
    "Treat selected predictors as categorical",
    predictors, default=default_cats
)

quantitative = [c for c in predictors if c not in categorical]

# Lists the quantitative and categorical predictors
st.write("Quantitative:", quantitative)
st.write("Categorical:", categorical)



st.subheader("Explore the data")
plot_type = st.radio("Plot type", ["Histogram", "Boxplot", "Scatterplot"])
plot_col = st.selectbox("Numeric variable", numeric, key="plot_col")

# Customize plot color
plot_color = st.color_picker("Choose plot color", "#1f77b4")

# Customize histogram bins
if plot_type == "Histogram":
    bins = st.slider("Number of bins", 5, 50, 20)

fig, ax = plt.subplots()
if plot_type == "Histogram":
    ax.hist(data[plot_col].dropna(), bins=bins, color=plot_color, edgecolor="black")
elif plot_type == "Boxplot":
    ax.boxplot(data[plot_col].dropna(),
               patch_artist=True, boxprops=dict(facecolor=plot_color))
else:
    ax.scatter(data[plot_col], data[response], alpha=0.5, color=plot_color)
    ax.set_ylabel(response)
ax.set_xlabel(plot_col)
ax.set_title(f"{plot_type}: {plot_col}")
st.pyplot(fig)
plt.close(fig)



st.subheader("Linear regression")
fit_vars = st.multiselect("Numeric model predictors", quantitative)
if st.button("Fit regression"):
    if not fit_vars:
        st.warning("Select at least one numeric predictor.")
    else:
        clean = data[[response] + fit_vars].dropna()
        X = clean[fit_vars]
        y = clean[response]
        model = LinearRegression().fit(X, y)
        predicted = model.predict(X)
        residuals = y.to_numpy() - predicted
        n, p = X.shape
        rss = np.sum(residuals**2)
        if n <= p + 1:
            st.error("Insufficient residual degrees of freedom.")
        else:
            st.session_state["fit"] = dict(
                model=model, X=X, y=y, predicted=predicted,
                residuals=residuals, rss=rss,
                rse=np.sqrt(rss/(n-p-1)),
                r2=r2_score(y, predicted)
            )
if "fit" in st.session_state:
    r = st.session_state["fit"]
    st.write("R-squared:", round(r["r2"], 4))
    st.write("RSS:", round(r["rss"], 2))
    st.write("RSE:", round(r["rse"], 2))
    st.write("Intercept:", r["model"].intercept_)
    st.dataframe(pd.DataFrame({
        "Predictor": r["X"].columns,
        "Coefficient": r["model"].coef_
    }))




st.subheader("Residual diagnostics")
if "fit" in st.session_state:
    r = st.session_state["fit"]
    c1, c2 = st.columns(2)
    fig1, ax1 = plt.subplots()
    ax1.scatter(r["predicted"], r["residuals"], alpha=0.5)
    ax1.axhline(0, color="black", linestyle="--")
    ax1.set(xlabel="Fitted", ylabel="Residual",
            title="Residuals versus fitted")
    c1.pyplot(fig1)
    plt.close(fig1)
    fig2, ax2 = plt.subplots()
    ax2.hist(r["residuals"], bins=20, edgecolor="black")
    ax2.set(title="Residual distribution", xlabel="Residual")
    c2.pyplot(fig2)
    plt.close(fig2)




st.subheader("Polynomial experiment")

if quantitative:
    poly_var = st.selectbox("Polynomial variable", quantitative)
    degree = st.slider("Degree", 1, 5, 2)
    sub = data[[response, poly_var]].dropna()
    X0 = sub[[poly_var]]
    y0 = sub[response]
    transformer = PolynomialFeatures(degree=degree, include_bias=False)
    Xp = transformer.fit_transform(X0)
    poly_model = LinearRegression().fit(Xp, y0)

    # Display the polynomial regression coefficients
    st.write("Intercept:", poly_model.intercept_)

    coefficient_table = pd.DataFrame({
        "Term": transformer.get_feature_names_out(),
        "Coefficient": poly_model.coef_
    })

    st.dataframe(coefficient_table)
    st.write("Training R-squared:", round(poly_model.score(Xp, y0), 4))
    st.write("Terms:", transformer.get_feature_names_out())

    grid = pd.DataFrame({
        poly_var: np.linspace(X0[poly_var].min(),
                             X0[poly_var].max(), 200)
    })

    fig, ax = plt.subplots()
    ax.scatter(X0[poly_var], y0, alpha=0.3)
    ax.plot(grid[poly_var], poly_model.predict(
        transformer.transform(grid)), color="black")
    ax.set(xlabel=poly_var, ylabel=response, title=f"Degree {degree}")
    st.pyplot(fig)
    plt.close(fig)


# Compare residuals for linear and polynomial models
st.subheader("Model Residual Comparison")

if quantitative:
    residual_model = st.selectbox(
        "Choose model for residual plot",
        ["Linear Regression", "Polynomial Regression"]
    )

    sub = data[[response, poly_var]].dropna()
    X0 = sub[[poly_var]]
    y0 = sub[response]

    if residual_model == "Linear Regression":
        selected_model = LinearRegression().fit(X0, y0)
        predictions = selected_model.predict(X0)

    else:
        transformer = PolynomialFeatures(
            degree=degree, include_bias=False
        )
        Xp = transformer.fit_transform(X0)
        selected_model = LinearRegression().fit(Xp, y0)
        predictions = selected_model.predict(Xp)

    residuals = y0.to_numpy() - predictions

    fig, ax = plt.subplots()
    ax.scatter(predictions, residuals, alpha=0.5)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("Residuals")
    ax.set_title(f"Residuals: {residual_model}")

    st.pyplot(fig)
    plt.close(fig)
    
    
    
    st.subheader("Categorical encoding")
if categorical and st.checkbox("Show dummy-variable preview"):
    raw = data[categorical].dropna().astype(str)
    enc = OneHotEncoder(drop="first", sparse_output=False)
    values = enc.fit_transform(raw)
    names = enc.get_feature_names_out(categorical)
    st.dataframe(pd.DataFrame(values, columns=names).head())
    for name, levels in zip(categorical, enc.categories_):
        st.write(f"{name}: reference category = {levels[0]}")



st.subheader("Interaction experiment")
if len(quantitative) >= 2:
    x1 = st.selectbox("First variable", quantitative, key="x1")
    x2 = st.selectbox("Second variable",
                      [x for x in quantitative if x != x1], key="x2")
    if st.button("Compare interaction"):
        sub = data[[response, x1, x2]].dropna().copy()
        sub["product"] = sub[x1] * sub[x2]
        y = sub[response]
        reduced = sub[[x1, x2]]
        full = sub[[x1, x2, "product"]]
        m0 = LinearRegression().fit(reduced, y)
        m1 = LinearRegression().fit(full, y)
        st.write("Additive R-squared:", m0.score(reduced, y))
        st.write("Interaction R-squared:", m1.score(full, y))
        st.dataframe(pd.DataFrame({
            "Term": full.columns, "Coefficient": m1.coef_
        }))




#hypothesis testing
st.subheader("Part 10: Coefficient Hypothesis Testing")
if "fit" not in st.session_state:
    st.info("Fit a quantitative regression model first.")
else:
    # Retrieve the last fitted regression model
    result = st.session_state["fit"]
    X = result["X"]
    y = result["y"]
    model = result["model"]
    # Let the user select a significance level
    # Allow users to customize significance level
    alpha = st.slider(
        "Significance level (alpha)",
        min_value=0.01,
        max_value=0.10,
        value=0.05,
        step=0.01
    )
    # Run the tests only when the button is pressed
    if st.button("Test coefficients"):
        # Number of observations and predictors
        n, p = X.shape
        # Residual degrees of freedom
        df = n - p - 1
        # Construct design matrix including intercept
        X_design = np.column_stack([
            np.ones(n),
            X.to_numpy(dtype=float)
        ])

        # Check degrees of freedom and matrix rank
        if n <= p + 1:
            st.error(
                "Insufficient observations for coefficient tests."
            )
        elif np.linalg.matrix_rank(X_design) < p + 1:
            st.error(
                "The design matrix does not have full column rank. "
                "Check for redundant or perfectly correlated predictors."
            )
        elif not np.isfinite(X_design).all():
            st.error("The predictor matrix contains invalid values.")
        else:
            # STEP 1: Obtain residuals and RSS
            predictions = model.predict(X)
            residuals = y.to_numpy() - predictions
            rss = np.sum(residuals ** 2)
            # STEP 2: Estimate error variance
            s_squared = rss / df

            # STEP 3: Coefficient covariance matrix
            XtX = X_design.T @ X_design
            XtX_inverse = np.linalg.inv(XtX)
            covariance_matrix = s_squared * XtX_inverse
            # STEP 4: Coefficient standard errors
            standard_errors = np.sqrt(
                np.diag(covariance_matrix)
            )
            # STEP 5: Estimated coefficients
            coefficients = np.concatenate([
                [model.intercept_],
                model.coef_.ravel()
            ])
            # STEP 6: Calculate t-statistics
            t_values = coefficients / standard_errors

            # STEP 7: Two-sided p-values
            p_values = 2 * stats.t.sf(
                np.abs(t_values),
                df=df
            )

            # STEP 8: Two-sided critical value
            t_critical = stats.t.ppf(
                1 - alpha / 2,
                df=df
            )
            # Build the coefficient table
            term_names = ["Intercept"] + list(X.columns)
            coefficient_table = pd.DataFrame({
                "Term": term_names,
                "Coefficient": coefficients,
                "Standard Error": standard_errors,
                "t-statistic": t_values,
                "p-value": p_values,
                "Decision": [
                    "Reject H0" if pv < alpha
                    else "Fail to reject H0"
                    for pv in p_values
                ]
            })
            # Save results so the plot can update independently
            st.session_state["t_test_results"] = {
                "table": coefficient_table,
                "df": df,
                "alpha": alpha,
                "t_critical": t_critical,
                "n": n,
                "p": p,
                "s_squared": s_squared
            }
    # DISPLAY THE RESULTS
    if "t_test_results" in st.session_state:
        test = st.session_state["t_test_results"]
        # Only show results for the alpha value used
        if test["alpha"] == alpha:
            st.write("Number of observations:", test["n"])
            st.write("Number of predictors:", test["p"])
            st.write("Residual degrees of freedom:", test["df"])
            st.write(
                "Estimated error variance:",
                round(test["s_squared"], 4)
            )
            st.write(
                "Critical t-values:",
                round(-test["t_critical"], 4),
                round(test["t_critical"], 4)
            )
            st.dataframe(
                test["table"].style.format({
                    "Coefficient": "{:.4f}",
                    "Standard Error": "{:.4f}",
                    "t-statistic": "{:.4f}",
                    "p-value": "{:.4g}"
                })
            )
            # VISUALIZE THE T-TEST
            st.subheader("Visualize an Individual Coefficient Test")
            selected_term = st.selectbox(
                "Select a coefficient",
                test["table"]["Term"].tolist()
            )
            selected_row = test["table"].loc[
                test["table"]["Term"] == selected_term
            ].iloc[0]

            observed_t = float(selected_row["t-statistic"])
            observed_p = float(selected_row["p-value"])

            df = test["df"]
            critical = test["t_critical"]
            # Set horizontal range for the graph
            limit = max(
                4,
                abs(observed_t) + 1,
                critical + 1
            )
            x_values = np.linspace(-limit, limit, 1000)
            y_values = stats.t.pdf(x_values, df=df)
            fig, ax = plt.subplots(figsize=(9, 5))
            # Plot Student's t-distribution
            ax.plot(
                x_values,
                y_values,
                color="navy",
                label=f"t-distribution (df={df})"
            )

            # Shade left rejection region
            ax.fill_between(
                x_values,
                y_values,
                where=x_values <= -critical,
                color="red",
                alpha=0.5
            )
            # Shade right rejection region
            ax.fill_between(
                x_values,
                y_values,
                where=x_values >= critical,
                color="red",
                alpha=0.5,
                label="Rejection regions"
            )
            # Critical limits
            ax.axvline(
                -critical,
                color="red",
                linestyle="--",
                label=f"Critical values: ±{critical:.3f}"
            )
            ax.axvline(
                critical,
                color="red",
                linestyle="--"
            )
            # Observed t-statistic
            ax.axvline(
                observed_t,
                color="green",
                linewidth=2.5,
                label=f"Observed t = {observed_t:.3f}"
            )
            ax.set_xlabel("t-statistic")
            ax.set_ylabel("Probability density")
            ax.set_title(
                f"Two-sided t-test: {selected_term}"
            )
            ax.legend()
            ax.grid(alpha=0.2)
            st.pyplot(fig)
            plt.close(fig)
            # STATISTICAL CONCLUSION
            st.write("H0: The population coefficient equals zero.")
            st.write("H1: The population coefficient is not zero.")

            st.write("Observed t:", round(observed_t, 4))
            st.write("Two-sided p-value:", round(observed_p, 6))

            if observed_p < alpha:
                st.success(
                    f"Reject H0 at alpha = {alpha}. "
                    "The coefficient is statistically different "
                    "from zero under the model assumptions."
                )
            else:
                st.info(
                    f"Fail to reject H0 at alpha = {alpha}. "
                    "There is insufficient evidence that the "
                    "coefficient differs from zero."
                )





# Part 11: Nested F-Tests
st.subheader("Part 11: Nested F-Tests")

# From Advertising Interaction Study Guide
def nested_f_test(y, X_reduced, X_full):
    reduced = LinearRegression().fit(X_reduced, y)
    full = LinearRegression().fit(X_full, y)

    rss_reduced = np.sum((y - reduced.predict(X_reduced))**2)
    rss_full = np.sum((y - full.predict(X_full))**2)

    n = len(y)
    p_reduced = X_reduced.shape[1]
    p_full = X_full.shape[1]
    q = p_full - p_reduced

    F = ((rss_reduced - rss_full) / q) / (rss_full / (n - p_full - 1))
    p_value = 1 - stats.f.cdf(F, q, n - p_full - 1)
    return F, p_value

# Select reduced and full model predictors
full_vars = st.multiselect("Full model predictors", predictors)
reduced_vars = st.multiselect("Reduced model predictors", full_vars)

if st.button("Compare nested models"):
    if not reduced_vars or len(full_vars) <= len(reduced_vars):
        st.warning("Select a reduced model and at least one additional predictor.")
    else:
        # Use the same data for both models
        clean = data[[response] + full_vars].dropna().copy()
        y = clean[response]

        # Convert categorical predictors to text
        for var in categorical:
            if var in clean.columns:
                clean[var] = clean[var].astype(str)

        # Create dummy variables
        X_full = pd.get_dummies(clean[full_vars], drop_first=True, dtype=float)
        X_reduced = pd.get_dummies(clean[reduced_vars], drop_first=True, dtype=float)
        n = len(y)
        p_full = X_full.shape[1]
        p_reduced = X_reduced.shape[1]

        # Check model degrees of freedom
        if n <= p_full + 1:
            st.error("Insufficient observations.")
        elif np.linalg.matrix_rank(
            np.column_stack([np.ones(n), X_full.to_numpy()])
        ) < p_full + 1:
            st.error("Redundant predictors in the full model.")
        else:
            F, p_value = nested_f_test(y, X_reduced, X_full)

            alpha = 0.05
            q = p_full - p_reduced
            df_full = n - p_full - 1
            F_critical = stats.f.ppf(1 - alpha, q, df_full)

            st.write("F-statistic:", round(F, 4))
            st.write("p-value:", p_value)
            st.write("Critical F-value:", round(F_critical, 4))

            if p_value < alpha:
                st.success("Reject H0: The full model improves the fit.")
            else:
                st.info("Fail to reject H0.")

            # F-distribution plot
            x_max = max(F + 2, F_critical + 2, 5)
            x_values = np.linspace(0.001, x_max, 1000)
            y_values = stats.f.pdf(x_values, q, df_full)

            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(x_values, y_values, color="navy")

            ax.fill_between(
                x_values, y_values,
                where=x_values >= F_critical,
                color="red", alpha=0.5,
                label="Rejection region"
            )

            ax.axvline(F_critical, color="red", linestyle="--",
                       label=f"Critical F = {F_critical:.3f}")
            ax.axvline(F, color="green", linewidth=2,
                       label=f"Observed F = {F:.3f}")

            ax.set_xlabel("F-statistic")
            ax.set_ylabel("Probability Density")
            ax.set_title("Nested Model F-Test")
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(alpha=0.5)

            st.pyplot(fig)
            plt.close(fig)




# Part 12: 10-Fold Cross-Validation
st.subheader("Part 12: 10-Fold Cross-Validation")

# Customize random state

random_state = st.slider("Random state", 0, 100, 1)
"""
random_state = (1)
"""


# Optional polynomial model
poly_options = ["None"] + [v for v in full_vars if v not in categorical]
poly_cv = st.selectbox("Polynomial predictor for CV", poly_options)

# Prepare training and testing predictors
def build_cv_data(train, test, variables, poly_var=None):
    train_columns = []
    test_columns = []

    for var in variables:
        if var in categorical:
            enc = OneHotEncoder(
                drop="first", sparse_output=False,
                handle_unknown="ignore"
            )
            train_values = enc.fit_transform(train[[var]].astype(str))
            test_values = enc.transform(test[[var]].astype(str))

        elif var == poly_var:
            poly = PolynomialFeatures(degree=2, include_bias=False)
            train_values = poly.fit_transform(train[[var]])
            test_values = poly.transform(test[[var]])

        else:
            train_values = train[[var]].to_numpy()
            test_values = test[[var]].to_numpy()

        train_columns.append(np.asarray(train_values).reshape(len(train), -1))
        test_columns.append(np.asarray(test_values).reshape(len(test), -1))

    X_train = np.column_stack(train_columns)
    X_test = np.column_stack(test_columns)

    return X_train, X_test


if st.button("Run 10-fold CV"):
    if not reduced_vars or len(full_vars) <= len(reduced_vars):
        st.warning("Select reduced and full models in Part 11 first.")
    else:
        # Use the same complete observations for every model
        clean = data[[response] + full_vars].dropna().copy()

        if len(clean) < 10:
            st.error("At least 10 observations are required.")
        else:
            # Define candidate models
            candidate_models = [
                ("Reduced Model", reduced_vars, None),
                ("Full Model", full_vars, None)
            ]

            if poly_cv != "None":
                candidate_models.append(
                    ("Polynomial Model", full_vars, poly_cv)
                )

            # Same folds for all models
            kf = KFold(n_splits=10, shuffle=True,
                       random_state=random_state)
            folds = list(kf.split(clean))

            results = []

            for name, variables, poly_var in candidate_models:
                fold_mse = []

                for train_index, test_index in folds:
                    train = clean.iloc[train_index]
                    test = clean.iloc[test_index]

                    X_train, X_test = build_cv_data(
                        train, test, variables, poly_var
                    )

                    y_train = train[response]
                    y_test = test[response]

                    model = LinearRegression().fit(X_train, y_train)
                    predictions = model.predict(X_test)

                    mse = mean_squared_error(y_test, predictions)
                    fold_mse.append(mse)

                results.append({
                    "Model": name,
                    "CV MSE": np.mean(fold_mse)
                })

            # Display comparison
            cv_table = pd.DataFrame(results)
            st.dataframe(cv_table)

            best_model = cv_table.loc[cv_table["CV MSE"].idxmin(), "Model"]
            st.write("Lowest CV MSE:", best_model)


# Part 13: Classification Algorithms
st.header("Classification Algorithms")
st.write("Use Logistic Regression or LDA to predict a categorical response.")



# Select classification algorithm
classification_method = st.selectbox(
    "Classification method",
    ["Logistic Regression", "LDA"]
)

# Select the response variable
class_response = st.selectbox(
    "Classification response variable",
    data.columns,
    index=len(data.columns) - 1
)

# Select predictors, excluding the response
class_candidates = [c for c in data.columns if c != class_response]

class_predictors = st.multiselect(
    "Classification predictors",
    class_candidates
)

# Display selected variables
st.write("Response:", class_response)
st.write("Predictors:", ", ".join(class_predictors))



# Fit the selected classification model
if st.button("Fit classification model"):
    if not class_predictors:
        st.warning("Select at least one predictor.")
    else:
        # Remove observations with missing values
        clean = data[[class_response] + class_predictors].dropna()

        X = clean[class_predictors]
        y = clean[class_response].astype(str)

        # Check for numeric predictors
        if not all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns):
            st.error("Classification predictors must be numeric.")
        elif y.nunique() < 2:
            st.error("The response must have at least two categories.")
        else:
            # Select and fit the model
            if classification_method == "Logistic Regression":
                class_model = LogisticRegression(
                    C=1e10, solver="lbfgs",
                    max_iter=10000, tol=1e-10
                )
            else:
                class_model = LDA()

            class_model.fit(X, y)

            # Calculate predicted classes and probabilities
            class_predictions = class_model.predict(X)
            class_probabilities = class_model.predict_proba(X)

            # Save results for display
            st.session_state["classification"] = {
                "model": class_model,
                "X": X,
                "y": y,
                "predictions": class_predictions,
                "probabilities": class_probabilities
            }


# Display classification results
if "classification" in st.session_state:
    result = st.session_state["classification"]

    model = result["model"]
    y = result["y"]
    predictions = result["predictions"]
    probabilities = result["probabilities"]

    st.subheader("Classification Results")

    # Display predicted classes and probabilities
    prediction_table = pd.DataFrame({
        "Actual": y.to_numpy(),
        "Predicted": predictions
    })

    for i, label in enumerate(model.classes_):
        prediction_table[f"Probability of {label}"] = probabilities[:, i]

    st.write("First 10 predictions:")
    st.dataframe(prediction_table.head(10))

    # Calculate accuracy
    accuracy = accuracy_score(y, predictions)
    st.write("Accuracy:", round(accuracy, 4))
    st.write("Error rate:", round(1 - accuracy, 4))

    # Confusion matrix
    cm = confusion_matrix(y, predictions, labels=model.classes_)

    st.write("Confusion Matrix:")
    st.dataframe(pd.DataFrame(
        cm,
        index=model.classes_,
        columns=model.classes_
    ))

    # Additional metrics for binary classification
    if len(model.classes_) == 2:
        positive_class = model.classes_[1]

        sensitivity = recall_score(
            y, predictions, pos_label=positive_class
        )

        specificity = recall_score(
            y, predictions, pos_label=model.classes_[0]
        )

        precision = precision_score(
            y, predictions, pos_label=positive_class,
            zero_division=0
        )

        st.write("Positive class:", positive_class)
        st.write("Sensitivity:", round(sensitivity, 4))
        st.write("Specificity:", round(specificity, 4))
        st.write("Precision:", round(precision, 4))
