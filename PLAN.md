## Dataset Migration Plan: LUME to TOFU

### 1. Dataset Changes
- Original: LUME dataset with format `{"id", "input", "output", "task"}`
- New: TOFU dataset with format `{"question", "answer"}`
- Source: HuggingFace datasets library `locuslab/TOFU`
  - Forget set: `forget01` split
  - Retain set: `retain99` split

### 2. Required Modifications

#### A. Dependencies
- Add `datasets` library to requirements.txt

#### B. Code Changes

1. **src/unlearn_loading/downloads.py**
   - Remove LUME dataset downloading
   - Implement HuggingFace dataset loading
   - Add train/validation splitting
   - Add format transformation
   - Generate numerical IDs

2. **src/unlearn_loading/dataset_loading.py**
   - Update data processing for new format
   - Map fields:
     - "question" -> "input"
     - "answer" -> "output"
   - Add ID generation
   - Add task labeling (0=retain, 1=forget)

3. **Directory Changes**
   - Remove `sample_ids` directory (splits generated dynamically)

### 3. Data Processing Flow
1. Load TOFU datasets using HuggingFace
2. Split each set into train/validation
3. Transform data format
4. Generate numerical IDs
5. Process through existing pipeline

### 4. Testing Plan
1. Verify data loading
2. Check format transformation
3. Validate train/val splits
4. Test ID generation
5. Verify unlearning pipeline compatibility

### 5. Model Integration: Llama-3.2-1B

#### 5.1. Model Changes
- Added support for Meta's Llama-3.2-1B model
- Model source: `meta-llama/Llama-3.2-1B`

#### 5.2. Code Modifications

1. **src/unlearn.py**
   - Added `Llama-3.2-1B` to model choices
   - Updated model loading logic to handle new model type

2. **src/unlearn_loading/downloads.py**
   - Modified `download_model_1B` to support both OLMo-1B and Llama-3.2-1B
   - Added model type parameter for model selection
   - Added device mapping and trust remote code flags for Llama model

#### 5.3. Usage
Run with: `python src/unlearn.py --model Llama-3.2-1B`