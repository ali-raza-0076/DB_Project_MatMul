# Verification Scripts

## Purpose

Test scripts to verify correctness of sparse matrix operations against scipy (gold standard).

---

## verify_addition.py

**What it does**: Verifies sparse matrix addition (A+B) is correct

**How it works**:
1. Loads matrices A and B from sorted CSV files
2. Computes reference result using scipy: `C_scipy = A + B`
3. Loads your implementation's result
4. Compares all entries position-by-position

**Output**: 
- ✓ PERFECT if all entries match
- ✗ ERRORS if mismatches found (shows first few mismatches)

**Usage**:
```python
python verification/verify_addition.py
```

---

## verify_multiplication.py

**What it does**: Verifies sparse matrix multiplication (A×B) is correct

**How it works**:
1. Loads matrices A and B
2. Computes reference: `C_scipy = A @ B` (scipy matrix multiplication)
3. Loads your implementation's result
4. Compares all 197K+ entries

**Output**:
- Shows entry counts (scipy vs yours)
- Reports any mismatches with positions
- Confirms 1-based indexing is working

**Usage**:
```python
python verification/verify_multiplication.py
```

---

## verify_different_matrices.py

**What it does**: Confirms matrices A and B are actually different

**Why needed**: Ensures you're not computing A×A when you want A×B

**Checks**:
- Compares first few entries of A and B
- Verifies different sparsity patterns
- Confirms multiplication result is for A×B, not A×A

---

## verify_1based.py

**What it does**: Verifies 1-based indexing is implemented correctly

**Checks**:
- Input files use indices 1 to N (not 0 to N-1)
- Output files use indices 1 to N
- Internal computation still produces correct results

---

## How Verification Works

All scripts follow same pattern:

1. **Load your data** (1-based indices in CSV)
2. **Convert to 0-based** (scipy requirement)
3. **Compute scipy reference** (gold standard)
4. **Compare** position-by-position
5. **Report** match/mismatch status

**Why scipy?** Industry-standard library, highly tested, trusted reference implementation.

---

## Running All Verifications

```bash
cd verification
python verify_addition.py
python verify_multiplication.py  
python verify_different_matrices.py
python verify_1based.py
```

All should report ✓ PERFECT or ✓ CORRECT.
