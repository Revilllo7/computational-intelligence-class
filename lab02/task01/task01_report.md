# Lab02

## Task01: Data Validator

`data_validator.py` is a Python script that validates and corrects errors in the Iris dataset CSV file using Pandas alongside self-made validation and imputation strategies.

***Usage***:
```bash
python3 data_validator.py
```

The script automatically processes `../data/iris_big_with_errors.csv` and generates two output files in the `output/` directory.

### Console output
```yaml
Validation complete
Rows processed: 1500
Malformed rows repaired: 3
Malformed rows skipped: 0
Total log entries: 110
Unresolved issues: 0
Corrected dataset: /root/io/computational-intelligence-class/lab02/task01/output/iris_big_corrected.csv
Validation log: /root/io/computational-intelligence-class/lab02/task01/output/validation_log.csv
```

### Output Files

**1. `iris_big_corrected.csv`**
- Cleaned and corrected dataset with all errors resolved
- 1500 rows with 5 columns: sepal length, sepal width, petal length, petal width, target_name
- All numeric values rounded to 2 decimal places
- Species standardized to: `setosa`, `versicolor`, `virginica`

**2. `validation_log.csv`**
- Detailed audit log with 110 entries documenting all detected issues
- Columns: line_number, row_index, column, original_value, error_type, suspected_problem, action_taken, resolved, resolution_method, final_value, notes

### Features Implemented

#### 1. Structure Validation
The script validates that each row has exactly 5 fields and attempts to repair malformed rows:

**Malformed Row Repair:**
- Detects rows with incorrect field count (typically 6 fields due to comma in numeric values)
- Attempts merge of adjacent tokens that form valid decimal numbers
- Example: `4|75|2.50|4.14|1.08|versicolor` → `4.75|2.50|4.14|1.08|versicolor`
- example lines in `iris_big_with_errors.csv`: 3 malformed rows at lines 751, 983, 1068

#### 2. Numeric Data Validation & Correction
**Issues Detected:**
- Non-numeric tokens (e.g., `"abc"`)
- Decimal comma separators (e.g., `"4,75"` → `4.75`)
- Out-of-range values outside [0, 15] (e.g., `999.99` sentinel values)
- Missing/empty numeric values

**Imputation Strategy:**
1. **KNN Imputation (Primary):** Uses K-Nearest Neighbors (K=5) based on other numeric features within the same species class
2. **Class Median (Fallback):** Uses median of the same species if KNN fails
3. **Global Median (Last Resort):** Uses overall column median if class-specific data unavailable

**Example Corrections:**
| Line | Column | Original | Error Type | Resolution | Final Value |
|------|--------|----------|------------|------------|-------------|
| 156 | sepal width | 999.99 | range_error | knn_class | 3.49 |
| 225 | sepal length | (empty) | missing_value | knn_class | 5.17 |
| 392 | petal length | abc | type_coercion_error | knn_class | 1.54 |

#### 3. Species Normalization & Prediction
**Normalization Handles:**
- Spelling variants: `versicolour` → `versicolor`
- Prefix additions: `iris-setosa`, `iris_versicolor` → `setosa`, `versicolor`
- Special characters: `virginica?`, `versi-colour` → `virginica`, `versicolor`
- Case variations: `Setosa`, `VERSICOLOR` → `setosa`, `versicolor`
- Whitespace issues: `" setosa "` → `setosa`

**Species Prediction:**
- For unknown/null species, uses KNN classifier trained on rows with valid species
- Predicts based on similarity of 4 numeric features to known specimens
- Successfully predicted 4 unknown species entries

#### 4. Float Precision Handling
All numeric values are rounded to 2 decimal places to avoid floating-point precision errors:
- Before: `3.4899999999999998`
- After: `3.49`

### Validation Log Examples

**Structure Error (Repaired):**
```csv
line_number,column,original_value,error_type,action_taken,resolved,notes
751,<row>,4|75|2.50|4.14|1.08|versicolor,structure_error,Repaired malformed row,yes,Merged split decimal tokens at positions 0/1
```

**Missing Value (Imputed):**
```csv
line_number,column,original_value,error_type,action_taken,resolved,resolution_method,final_value
21,sepal length (cm),,missing_value,Imputed numeric value,yes,knn_class,5.37
```

**Species Typo (Normalized):**
```csv
line_number,column,original_value,error_type,action_taken,resolved,resolution_method,final_value
204,target_name,iris_versicolor,species_typo,Normalized to canonical species,yes,alias_normalization,versicolor
```

### Error Categories Summary

| Error Type | Count | Resolved |
|------------|-------|----------|
| missing_value | 65 | 65 ✓ |
| range_error | 19 | 19 ✓ |
| species_typo | 8 | 8 ✓ |
| type_coercion_error | 7 | 7 ✓ |
| species_unknown | 4 | 4 ✓ |
| species_predicted | 4 | 4 ✓ |
| structure_error | 3 | 3 ✓ |
| **Total** | **110** | **110 ✓** |

> [!NOTE]
> Species unknown and species predicted are separate, but double themselves. The 4 species unknown entries were all successfully predicted, so they are counted in both categories. This is because they represent two distinct issues: the initial unknown status and the subsequent successful prediction.

### Technical Implementation

**Key Functions:**
- `parse_input_file()` - Structure validation and malformed row repair
- `clean_numeric_columns()` - Numeric parsing, coercion, and range validation
- `clean_species()` - Species normalization using alias dictionary
- `impute_numeric_columns()` - Smart imputation with KNN → median fallback
- `infer_unknown_species()` - KNN-based species prediction
- `reconcile_pending_logs()` - Updates log entries to reflect final resolution status

**Libraries Used:**
- `pandas` - DataFrame operations and CSV handling
- `numpy` - Numeric operations and KNN distance calculations
- `csv` - Low-level CSV parsing for structure validation
- `re` - Regular expressions for text normalization

### Code Quality Features

- Comprehensive error logging with line-level traceability
- Non-destructive validation (original file unchanged)
- Deterministic K-NN with fixed random state for reproducibility
- Type-safe implementation with Python type hints
- Graceful degradation (KNN → median → global fallback)
- Detailed console summary of validation statistics

> [!NOTE]
> The script preserves source line numbers throughout processing, allowing each correction in the validation log to be traced back to the exact line in the original input file.
