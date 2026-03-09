from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR.parent / "data" / "iris_big_with_errors.csv"
OUTPUT_DIR = BASE_DIR / "output"
CORRECTED_FILE = OUTPUT_DIR / "iris_big_corrected.csv"
LOG_FILE = OUTPUT_DIR / "validation_log.csv"

EXPECTED_COLUMNS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "target_name",
]
NUMERIC_COLUMNS = EXPECTED_COLUMNS[:4]
CANONICAL_SPECIES = {"setosa", "versicolor", "virginica"}

# Numeric range given from the task. 0.0 rubs me the wrong way for width and petals, but
# What if a petal doesn't have a sepal at all? Do we allow 0 for sepals only?
# Well, what if the flower got damaged and lost all petals and it's just sepals remaining?
NUMERIC_MIN = 0.0
NUMERIC_MAX = 15.0
K_NEIGHBORS = 5

# Common spelling/format variants normalized to canonical target labels.
# Ignores -, spaces and special characters like (), ?, and upper/lower case
SPECIES_ALIASES = {
    "setosa": "setosa",
    "irissetosa": "setosa",
    "irisetosa": "setosa",
    "versicolor": "versicolor",
    "versicolour": "versicolor",
    "versicolr": "versicolor",
    "irisversicolor": "versicolor",
    "irisversicolour": "versicolor",
    "versicolourr": "versicolor",
    "irisversicolourr": "versicolor",
    "virginica": "virginica",
    "irisvirginica": "virginica",
}

LOG_COLUMNS = [
    "line_number",
    "row_index",
    "column",
    "original_value",
    "error_type",
    "suspected_problem",
    "action_taken",
    "resolved",
    "resolution_method",
    "final_value",
    "notes",
]



def create_log_entry(
    line_number: object,
    row_index: object,
    column: str,
    original_value: str,
    error_type: str,
    suspected_problem: str,
    action_taken: str,
    resolved: bool,
    resolution_method: str,
    final_value: str,
    notes: str = "",
) -> dict:
    return {
        "line_number": line_number,
        "row_index": row_index,
        "column": column,
        "original_value": original_value,
        "error_type": error_type,
        "suspected_problem": suspected_problem,
        "action_taken": action_taken,
        "resolved": "yes" if resolved else "no",
        "resolution_method": resolution_method,
        "final_value": final_value,
        "notes": notes,
    }

def normalize_header_name(value: str) -> str:
    return value.strip().strip('"')

def safe_int(value: object) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return -1

# Parse CSV with structure checks and preserve source line numbers.
def parse_input_file() -> tuple[pd.DataFrame, list[dict], int, int]:
    logs: list[dict] = []
    records: list[dict[str, object]] = []
    malformed_rows = 0
    repaired_rows = 0

    with INPUT_FILE.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise ValueError("Input CSV is empty.")


        # Normmalize header by stripping whitespace and quotes, then compare to expected columns.
        normalized_header = [normalize_header_name(col) for col in header]
        if normalized_header != EXPECTED_COLUMNS:
            logs.append(
                create_log_entry(
                    line_number=1,
                    row_index=None,
                    column="<header>",
                    original_value=",".join(header),
                    error_type="structure_error",
                    suspected_problem="Unexpected column names/order",
                    action_taken="Loaded using expected schema mapping by position",
                    resolved=len(normalized_header) == len(EXPECTED_COLUMNS),
                    resolution_method="schema_position_mapping",
                    final_value=",".join(normalized_header),
                )
            )


        # Check column number and attempt repair for (,) comma corrupting .csv
        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(EXPECTED_COLUMNS):
                repaired_row, repair_reason = try_repair_malformed_row(row)
                if repaired_row is not None:
                    repaired_rows += 1
                    logs.append(
                        create_log_entry(
                            line_number=line_number,
                            row_index=None,
                            column="<row>",
                            original_value="|".join(row),
                            error_type="structure_error",
                            suspected_problem=(
                                f"Expected {len(EXPECTED_COLUMNS)} fields, got {len(row)}"
                            ),
                            action_taken="Repaired malformed row",
                            resolved=True,
                            resolution_method="row_repair",
                            final_value="|".join(repaired_row),
                            notes=repair_reason,
                        )
                    )
                    row = repaired_row

                # If repair succeeded but still doesn't match expected columns, log as unresolved structure error
                else:
                    malformed_rows += 1
                    logs.append(
                        create_log_entry(
                            line_number=line_number,
                            row_index=None,
                            column="<row>",
                            original_value="|".join(row),
                            error_type="structure_error",
                            suspected_problem=(
                                f"Expected {len(EXPECTED_COLUMNS)} fields, got {len(row)}"
                            ),
                            action_taken="Skipped malformed row",
                            resolved=False,
                            resolution_method="skip_row",
                            final_value="",
                            notes="Likely delimiter/comma corruption in row",
                        )
                    )
                    continue

            record: dict[str, object] = {
                EXPECTED_COLUMNS[idx]: row[idx] for idx in range(len(EXPECTED_COLUMNS))
            }
            record["source_line"] = line_number
            records.append(record)

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("No valid rows found after structure validation.")

    return df, logs, malformed_rows, repaired_rows


# Detect if two adjacent fields could be a split decimal number due to comma corruption, e.g. "5,1" -> "5.1"
def _can_be_decimal_pair(left: str, right: str) -> bool:
    return bool(re.fullmatch(r"[-+]?\d+", left.strip())) and bool(
        re.fullmatch(r"\d+", right.strip())
    )


# Detect if a value looks like a species name, allowing for common typos and ignoring non-alphabetic characters.
def _looks_like_species(value: str) -> bool:
    normalized = re.sub(r"[^a-z]", "", value.strip().lower())
    if normalized.startswith("iris"):
        normalized = normalized[4:]
    return normalized in CANONICAL_SPECIES or normalized in SPECIES_ALIASES


# Attempt repair for rows with one extra split field from decimal comma corruption.
def try_repair_malformed_row(row: list[str]) -> tuple[list[str] | None, str]:
    """Attempt repair for rows with one extra split field from decimal comma corruption."""
    if len(row) != len(EXPECTED_COLUMNS) + 1:
        return None, "Unsupported malformed shape"

    # Try all adjacent merges and accept the first candidate with numeric first 4 and plausible species.
    for merge_pos in range(len(row) - 1):
        left = row[merge_pos].strip().strip('"')
        right = row[merge_pos + 1].strip().strip('"')
        if not _can_be_decimal_pair(left, right):
            continue

        merged = f"{left}.{right}"
        candidate = row[:merge_pos] + [merged] + row[merge_pos + 2 :]
        if len(candidate) != len(EXPECTED_COLUMNS):
            continue

        numeric_ok = True
        for token in candidate[:4]:
            parsed, _, _ = parse_numeric_value(str(token))
            if parsed is None:
                numeric_ok = False
                break

        species_ok = _looks_like_species(candidate[4])
        if numeric_ok and species_ok:
            return candidate, f"Merged split decimal tokens at positions {merge_pos}/{merge_pos + 1}"

    return None, "No safe merge candidate found"


# Normalize species labels to canonical form, log issues, and mark unknowns for later KNN check.
def clean_species(df: pd.DataFrame, logs: list[dict]) -> None:
    normalized_species: list[str | None] = []

    for idx, raw_value in df["target_name"].fillna("").astype(str).items():
        line_number = safe_int(df.at[idx, "source_line"])
        original = raw_value
        stripped = raw_value.strip().strip('"').lower()

        if not stripped:
            normalized_species.append(None)
            logs.append(
                create_log_entry(
                    line_number,
                    idx,
                    "target_name",
                    original,
                    "species_unknown",
                    "Missing species label",
                    "Marked as unknown for later KNN inference",
                    False,
                    "pending_knn",
                    "",
                )
            )
            continue

        folded = re.sub(r"[^a-z]", "", stripped)
        if folded.startswith("iris"):
            folded = folded[4:]

        mapped = SPECIES_ALIASES.get(folded)
        if mapped is None:
            normalized_species.append(None)
            logs.append(
                create_log_entry(
                    line_number,
                    idx,
                    "target_name",
                    original,
                    "species_unknown",
                    "Unrecognized species value",
                    "Marked as unknown for later KNN inference",
                    False,
                    "pending_knn",
                    "",
                )
            )
        else:
            normalized_species.append(mapped)
            if mapped != stripped:
                logs.append(
                    create_log_entry(
                        line_number,
                        idx,
                        "target_name",
                        original,
                        "species_typo",
                        "Species variant/typo detected",
                        "Normalized to canonical species",
                        True,
                        "alias_normalization",
                        mapped,
                    )
                )

    df["target_name"] = normalized_species


def parse_numeric_value(text: str) -> tuple[float | None, str | None, str | None]:
    # Return numeric value and optional (problem, action) for logging.
    raw = text.strip().strip('"')
    if raw == "":
        return None, "Missing numeric value", "Marked as missing for imputation"

    adjusted = raw
    if re.fullmatch(r"[-+]?\d+,\d+", raw):
        adjusted = raw.replace(",", ".")
        return_value = float(adjusted)
        return (
            return_value,
            "Comma used as decimal separator",
            "Converted comma decimal to dot",
        )

    try:
        return float(adjusted), None, None
    except ValueError:
        return None, "Non-numeric token", "Marked as missing for imputation"


# Clean numeric columns with type coercion, range checks, and logging of issues. Mark invalid/missing values as NaN for later imputation.
def clean_numeric_columns(df: pd.DataFrame, logs: list[dict]) -> None:
    for column in NUMERIC_COLUMNS:
        cleaned_values: list[float] = []
        for idx, raw_value in df[column].fillna("").astype(str).items():
            line_number = safe_int(df.at[idx, "source_line"])
            numeric_value, problem, action = parse_numeric_value(raw_value)

            if problem is not None:
                logs.append(
                    create_log_entry(
                        line_number,
                        idx,
                        column,
                        raw_value,
                        "type_coercion_error" if "Non-numeric" in problem else "missing_value",
                        problem,
                        action or "",
                        False,
                        "pending_imputation",
                        "",
                    )
                )

            if numeric_value is not None and not (NUMERIC_MIN <= numeric_value <= NUMERIC_MAX):
                logs.append(
                    create_log_entry(
                        line_number,
                        idx,
                        column,
                        raw_value,
                        "range_error",
                        f"Value outside range [{NUMERIC_MIN}, {NUMERIC_MAX}]",
                        "Set to missing for imputation",
                        False,
                        "pending_imputation",
                        "",
                    )
                )
                numeric_value = None

            cleaned_values.append(np.nan if numeric_value is None else float(numeric_value))

        df[column] = cleaned_values


# KNN imputation for a single cell, using available numeric features and optionally restricting to same species.
def knn_impute_for_cell(
    df: pd.DataFrame,
    row_idx: int,
    target_column: str,
    species: str | None,
    k: int,
) -> tuple[float | None, str]:
    feature_cols = [col for col in NUMERIC_COLUMNS if col != target_column]
    row_features = df.loc[row_idx][feature_cols]
    available_features = [col for col in feature_cols if pd.notna(row_features[col])]

    if not available_features:
        return None, "No usable features for KNN"

    candidate_mask = df[target_column].notna()
    if species in CANONICAL_SPECIES:
        candidate_mask = candidate_mask & (df["target_name"] == species)

    candidates = df.loc[candidate_mask].copy()
    candidates = candidates.drop(index=row_idx, errors="ignore")
    if candidates.empty:
        return None, "No candidates for KNN"

    for col in available_features:
        candidates = candidates[candidates[col].notna()]
    if candidates.empty:
        return None, "Candidates missing required feature values"

    candidate_matrix = candidates[available_features].astype(float).to_numpy()
    target_vector = row_features[available_features].astype(float).to_numpy()
    distances = np.sqrt(((candidate_matrix - target_vector) ** 2).sum(axis=1))
    nearest_count = max(1, min(k, len(candidates)))
    nearest_positions = np.argsort(distances)[:nearest_count]
    nearest = candidates.iloc[nearest_positions]
    value = round(float(nearest[target_column].mean()), 2)
    return value, ""


# Impute missing numeric values Fall back to class median or global median if KNN fails.
def impute_numeric_columns(df: pd.DataFrame, logs: list[dict]) -> None:
    for row_idx_obj in df.index:
        row_idx = int(row_idx_obj)
        raw_species = df.at[row_idx, "target_name"]
        species = str(raw_species) if pd.notna(raw_species) else None
        line_number = safe_int(df.at[row_idx, "source_line"])
        for column in NUMERIC_COLUMNS:
            if pd.notna(df.at[row_idx, column]):
                continue

            imputed_value, reason = knn_impute_for_cell(df, row_idx, column, species, K_NEIGHBORS)
            method = "knn_class"

            if imputed_value is None and species in CANONICAL_SPECIES:
                class_median = df.loc[df["target_name"] == species, column].median(skipna=True)
                if pd.notna(class_median):
                    imputed_value = round(float(class_median), 2)
                    method = "median_class"

            if imputed_value is None:
                global_median = df[column].median(skipna=True)
                if pd.notna(global_median):
                    imputed_value = round(float(global_median), 2)
                    method = "median_global"

            if imputed_value is None:
                logs.append(
                    create_log_entry(
                        line_number,
                        row_idx,
                        column,
                        "",
                        "missing_value",
                        "Unable to impute numeric value",
                        "Left unresolved",
                        False,
                        "unresolved",
                        "",
                        notes=reason,
                    )
                )
                continue

            df.at[row_idx, column] = imputed_value
            logs.append(
                create_log_entry(
                    line_number,
                    row_idx,
                    column,
                    "",
                    "missing_value",
                    "Numeric value missing after cleaning",
                    "Imputed numeric value",
                    True,
                    method,
                    f"{imputed_value:.2f}",
                    notes=reason,
                )
            )


# KNN prediction for unknown species based on numeric features. Restrict to same species if possible, otherwise use all data.
def knn_predict_species(df: pd.DataFrame, row_idx: int, k: int) -> tuple[str | None, str]:
    train = df[df["target_name"].isin(CANONICAL_SPECIES)].copy()
    train = train.dropna(subset=NUMERIC_COLUMNS)
    if train.empty:
        return None, "No training rows with canonical species"

    target_features = df.loc[row_idx][NUMERIC_COLUMNS]
    if target_features.isna().any():
        return None, "Missing numeric features for species prediction"

    train_matrix = train[NUMERIC_COLUMNS].astype(float).to_numpy()
    target_vector = target_features.astype(float).to_numpy()
    distances = np.sqrt(((train_matrix - target_vector) ** 2).sum(axis=1))
    nearest_count = max(1, min(k, len(train)))
    nearest_positions = np.argsort(distances)[:nearest_count]
    nearest = train.iloc[nearest_positions]
    predicted = nearest["target_name"].mode().iat[0]
    return str(predicted), ""


def infer_unknown_species(df: pd.DataFrame, logs: list[dict]) -> None:
    for row_idx_obj in df.index[df["target_name"].isna()]:
        row_idx = int(row_idx_obj)
        line_number = safe_int(df.at[row_idx, "source_line"])
        predicted, reason = knn_predict_species(df, row_idx, K_NEIGHBORS)
        if predicted is None:
            logs.append(
                create_log_entry(
                    line_number,
                    row_idx,
                    "target_name",
                    "",
                    "species_unknown",
                    "Species unresolved after normalization",
                    "Left unresolved",
                    False,
                    "unresolved",
                    "",
                    notes=reason,
                )
            )
            continue

        df.at[row_idx, "target_name"] = predicted
        logs.append(
            create_log_entry(
                line_number,
                row_idx,
                "target_name",
                "",
                "species_predicted",
                "Species unknown after normalization",
                "Predicted species from numeric neighbors",
                True,
                "knn_species",
                predicted,
            )
        )


# Check the schema and no unknown/null values remain. Log any remaining issues as unresolved.
def final_integrity_checks(df: pd.DataFrame, logs: list[dict]) -> None:
    if list(df.columns[:5]) != EXPECTED_COLUMNS:
        logs.append(
            create_log_entry(
                line_number=None,
                row_index=None,
                column="<dataset>",
                original_value=",".join(df.columns),
                error_type="structure_error",
                suspected_problem="Final output columns do not match expected schema",
                action_taken="No automatic fix",
                resolved=False,
                resolution_method="none",
                final_value=",".join(EXPECTED_COLUMNS),
            )
        )

    unknown_species = ~df["target_name"].isin(CANONICAL_SPECIES)
    if unknown_species.any():
        for idx in df[unknown_species].index:
            logs.append(
                create_log_entry(
                    line_number=safe_int(df.at[idx, "source_line"]),
                    row_index=idx,
                    column="target_name",
                    original_value=str(df.at[idx, "target_name"]),
                    error_type="species_unknown",
                    suspected_problem="Non-canonical species remains in output",
                    action_taken="No automatic fix",
                    resolved=False,
                    resolution_method="none",
                    final_value=str(df.at[idx, "target_name"]),
                )
            )


# Update pending entries when final dataframe already contains a valid value.
# This allows us to resolve some issues that were marked as pending during cleaning if they got fixed later
# by imputation or species inference. For example, if a numeric value was missing and marked as pending imputation, 
# but then got imputed successfully, we can go back and mark that issue as resolved with the imputed value.
def reconcile_pending_logs(df: pd.DataFrame, logs: list[dict]) -> list[dict]:
    reconciled: list[dict] = []
    for entry in logs:
        updated = dict(entry)
        if updated.get("resolved") != "no":
            reconciled.append(updated)
            continue

        method = str(updated.get("resolution_method", ""))
        row_index = updated.get("row_index")
        column = str(updated.get("column", ""))

        if row_index in (None, ""):
            reconciled.append(updated)
            continue

        try:
            idx = int(float(row_index))
        except (TypeError, ValueError):
            reconciled.append(updated)
            continue

        if idx not in df.index:
            reconciled.append(updated)
            continue

        if method == "pending_imputation" and column in NUMERIC_COLUMNS:
            if pd.notna(df.at[idx, column]):
                numeric_value = df.at[idx, column]
                try:
                    formatted_value = f"{pd.to_numeric(numeric_value):.2f}"
                except (TypeError, ValueError):
                    formatted_value = str(numeric_value)
                updated["resolved"] = "yes"
                updated["action_taken"] = "Resolved during imputation phase"
                updated["resolution_method"] = "resolved_post_imputation"
                updated["final_value"] = formatted_value

        if method == "pending_knn" and column == "target_name":
            final_species = df.at[idx, "target_name"]
            if pd.notna(final_species) and str(final_species) in CANONICAL_SPECIES:
                updated["resolved"] = "yes"
                updated["action_taken"] = "Resolved during species inference"
                updated["resolution_method"] = "resolved_post_species"
                updated["final_value"] = str(final_species)

        reconciled.append(updated)

    return reconciled


# Save the corrected dataset and validation log to CSV files. Print a summary of the validation results.
def save_outputs(
    df: pd.DataFrame, logs: list[dict], malformed_rows: int, repaired_rows: int
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    corrected = df[EXPECTED_COLUMNS].copy()
    # Round numeric columns to 2 decimal places to avoid float precision issues
    for col in NUMERIC_COLUMNS:
        corrected[col] = corrected[col].round(2)
    corrected.to_csv(CORRECTED_FILE, index=False)

    final_logs = reconcile_pending_logs(df, logs)
    log_df = pd.DataFrame(final_logs, columns=LOG_COLUMNS)
    log_df.to_csv(LOG_FILE, index=False)

    unresolved = int((log_df["resolved"] == "no").sum()) if not log_df.empty else 0
    print("Validation complete")
    print(f"Rows processed: {len(df)}")
    print(f"Malformed rows repaired: {repaired_rows}")
    print(f"Malformed rows skipped: {malformed_rows}")
    print(f"Total log entries: {len(log_df)}")
    print(f"Unresolved issues: {unresolved}")
    print(f"Corrected dataset: {CORRECTED_FILE}")
    print(f"Validation log: {LOG_FILE}")


def main() -> None:
    df, logs, malformed_rows, repaired_rows = parse_input_file()
    clean_species(df, logs)
    clean_numeric_columns(df, logs)
    impute_numeric_columns(df, logs)
    infer_unknown_species(df, logs)
    final_integrity_checks(df, logs)
    save_outputs(df, logs, malformed_rows, repaired_rows)


if __name__ == "__main__":
    main()