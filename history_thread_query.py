
import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI


# ------------------------
# 1. Load and enrich DataFrame
# ------------------------

def load_and_enrich_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Fill missing board info with 'NoBoard'
    df['examBoardTitle'] = df['examBoardTitle'].fillna("NoBoard")

    # Step 1: Group and aggregate lesson-level fields per unitId and thread
    agg_fields = {
        'priorKnowledgeRequirements': lambda x: "\n".join(x.dropna().unique()),
        'pupilLessonOutcome': lambda x: "\n".join(x.dropna().unique()),
        'keyLearningPoints': lambda x: "\n".join(x.dropna().unique())
    }
    meta_fields = [
        'unitId', 'unitTitle', 'yearTitle', 'keyStageTitle', 'examBoardTitle',
        'threadTitle', 'threadDescription', 'unitDescription', 'unitOrder'
    ]
    df_grouped = df.groupby(['unitId', 'threadTitle'], as_index=False).agg({
        **{k: 'first' for k in meta_fields},
        **agg_fields
    })

    # Step 2: Extract numeric year group
    df_grouped['yearGroup'] = df_grouped['yearTitle'].str.extract(r'(\d+)').astype(int)

    # Step 3: Build exam-board-aware thread sequences (or fallback to NoBoard-only)
    all_sequences = []
    unique_boards = df_grouped['examBoardTitle'].unique()
    real_boards = [b for b in unique_boards if b != "NoBoard"]

    if real_boards:
        for board in real_boards:
            board_df = df_grouped[
                (df_grouped['examBoardTitle'] == board) |
                (df_grouped['examBoardTitle'] == "NoBoard")
            ].copy()

            board_df['effectiveBoard'] = board
            board_df = board_df.sort_values(by=['threadTitle', 'yearGroup', 'unitOrder'])

            board_df['prevUnitId'] = board_df.groupby(['threadTitle', 'effectiveBoard'])['unitId'].shift(1)
            board_df['nextUnitId'] = board_df.groupby(['threadTitle', 'effectiveBoard'])['unitId'].shift(-1)
            board_df['prevExamBoard'] = board_df.groupby(['threadTitle', 'effectiveBoard'])['examBoardTitle'].shift(1)
            board_df['nextExamBoard'] = board_df.groupby(['threadTitle', 'effectiveBoard'])['examBoardTitle'].shift(-1)

            all_sequences.append(board_df)
    else:
        # fallback: everything is "NoBoard"
        board_df = df_grouped.copy()
        board_df['effectiveBoard'] = "NoBoard"
        board_df = board_df.sort_values(by=['threadTitle', 'yearGroup', 'unitOrder'])

        board_df['prevUnitId'] = board_df.groupby(['threadTitle'])['unitId'].shift(1)
        board_df['nextUnitId'] = board_df.groupby(['threadTitle'])['unitId'].shift(-1)
        board_df['prevExamBoard'] = board_df.groupby(['threadTitle'])['examBoardTitle'].shift(1)
        board_df['nextExamBoard'] = board_df.groupby(['threadTitle'])['examBoardTitle'].shift(-1)

        all_sequences.append(board_df)

    df_enriched = pd.concat(all_sequences, ignore_index=True)

    # Step 4: Merge in metadata for prev/next units
    unit_lookup = df_grouped.drop_duplicates(subset='unitId')[
        ['unitId', 'unitTitle', 'yearTitle', 'keyStageTitle', 'unitDescription']
    ].copy()
    unit_lookup.columns = ['lookupUnitId', 'lookupTitle', 'lookupYear', 'lookupKeyStage', 'lookupDesc']

    df_enriched = pd.merge(
        df_enriched,
        unit_lookup,
        left_on='prevUnitId',
        right_on='lookupUnitId',
        how='left'
    ).rename(columns={
        'lookupTitle': 'prevUnitTitle',
        'lookupYear': 'prevUnitYear',
        'lookupKeyStage': 'prevKeyStage',
        'lookupDesc': 'prevUnitDesc'
    }).drop(columns=['lookupUnitId'])

    df_enriched = pd.merge(
        df_enriched,
        unit_lookup,
        left_on='nextUnitId',
        right_on='lookupUnitId',
        how='left'
    ).rename(columns={
        'lookupTitle': 'nextUnitTitle',
        'lookupYear': 'nextUnitYear',
        'lookupKeyStage': 'nextKeyStage',
        'lookupDesc': 'nextUnitDesc'
    }).drop(columns=['lookupUnitId'])

    return df_enriched



# ------------------------
# 2. Paragraph 1 construction
# ------------------------

def extract_key_stage(ks_title):
    """Extracts the numeric part of 'Key Stage x' and returns 'KSx'."""
    if pd.isna(ks_title):
        return "KS?"
    match = pd.Series(ks_title).str.extract(r'(\d+)').iloc[0, 0]
    return f"KS{match}" if pd.notna(match) else "KS?"

def extract_year_group(year_title):
    """Extracts the numeric part of 'Year x' and returns 'Yx'."""
    if pd.isna(year_title):
        return "Y?"
    match = pd.Series(year_title).str.extract(r'(\d+)').iloc[0, 0]
    return f"Y{match}" if pd.notna(match) else "Y?"

def format_unit_string(unit_id, title, year_title, key_stage_title, exam_board_title=None):
    """Formats a single unit line for paragraph1."""
    if pd.isna(unit_id) or pd.isna(title) or pd.isna(year_title) or pd.isna(key_stage_title):
        return "N/A"

    ks = extract_key_stage(key_stage_title)
    yg = extract_year_group(year_title)
    eb = f"({exam_board_title}) " if exam_board_title and exam_board_title != "NoBoard" else ""

    return f"{int(unit_id)} {eb}{title} ({ks}, {yg})"

def make_paragraph1(row):
    thread = row['threadTitle']
    prev_str = format_unit_string(
        row['prevUnitId'],
        row['prevUnitTitle'],
        row['prevUnitYear'],
        row['prevKeyStage'],
        row.get('prevExamBoard')
    )
    curr_str = format_unit_string(
        row['unitId'],
        row['unitTitle'],
        row['yearTitle'],
        row['keyStageTitle'],
        row.get('examBoardTitle')
    )
    next_str = format_unit_string(
        row['nextUnitId'],
        row['nextUnitTitle'],
        row['nextUnitYear'],
        row['nextKeyStage'],
        row.get('nextExamBoard')
    )

    return f"""Thread: {thread}
Previous Unit: {prev_str}
Current Unit: {curr_str}
Next Unit: {next_str}"""


# ------------------------
# 3. Prompt construction
# ------------------------

def format_unit(uid, title, year):
    if pd.isna(uid):
        return "N/A"
    return f"{int(uid)} {title} (KS3, {year})"


def construct_prompt(row_group, prior_progression="") -> str:
    unit = row_group.iloc[0]

    thread_title = unit['threadTitle']
    thread_desc = unit['threadDescription']

    previous_unit = format_unit(unit['prevUnitId'], unit['prevUnitTitle'], unit['prevUnitYear'])
    previous_unit_desc = unit['prevUnitDesc']
    
    current_unit = format_unit(unit['unitId'], unit['unitTitle'], unit['yearTitle'])
    current_unit_desc = unit['unitDescription']
    
    next_unit = format_unit(unit['nextUnitId'], unit['nextUnitTitle'], unit['nextUnitYear'])
    next_unit_desc = unit['nextUnitDesc']

    prior_knowledge = unit['priorKnowledgeRequirements']
    pupil_outcomes = unit['pupilLessonOutcome']
    klps = unit['keyLearningPoints']
    
    if prior_progression == "":
        prior_progression = "None."

    prompt = f"""
TASK: You are going to write a THREAD PROGRESSION STATEMENT that explains how 
a thread progresses across Oak National Academy's history curriculum. 

A THREAD shows how units connect to one another, helping to group and link 
conceptually connected units, within and across different year groups.

AUDIENCE: This is designed for teachers and curriculum designers working in 
schools in the UK.  You should write in British English.  Assume the reader is 
familiar with Oak's curriculum design principles and concept of THREADS but 
doesn't know anything about the individual threads or units.


DEFINITION OF THREADS: 

In history, threads are long-term organising structures made up of broad 
collections of important substantive concepts that span the entire curriculum. 
Substantive concepts are the powerful, recurring ideas that help pupils make 
sense of historical content across time and place. Substantive concepts 
represent abstract, transferable ideas such as empire, parliament, slavery, 
peasant, revolution, or middle class. 

These concepts:
- Span the curriculum, appearing in multiple topics from Key Stage 1 through to 
Key Stage 4.
- Accrue meaning over time, deepening as students revisit them in new historical 
contexts (e.g., understanding “settlement” through Viking invasions, Norman 
conquest, and later immigration in the era after the Second World War).
- Gain richness through stories and concrete examples, not dictionary definitions; 
their meaning lives in the interplay between narrative, context, and prior 
knowledge.
- Support historical thinking by enabling pupils to draw comparisons, spot 
continuity and change, and recognise patterns across historical periods.

Substantive concepts do not have fixed definitions. Their meanings are 
historically contingent and develop as pupils engage with multiple historical 
examples. The concept of “revolution,” for instance, changes shape between the 
French, Industrial, and Russian revolutions but retains a useful core that 
supports understanding.

Threads link curriculum units to one another to achieve coherence and 
progression. They group and link conceptually connected units, within and 
across different year groups as appropriate. The units associated with each 
thread are purposefully and coherently linked to build knowledge, skills and 
understanding linked to each thread over time.

In the History curriculum, there are 5 such threads:
- Power, government and religion
- Invasion, migration and settlement
- Warfare and conflict
- Empire, persecution and resistance
- Trade, ideas and communication


INPUT PROVIDED:

Thread: {thread_title}
Thread description:\n{thread_desc}

Current Unit: {current_unit}
Current Unit Description:\n{current_unit_desc}

Prior knowledge:\n{prior_knowledge}
Pupil lesson outcomes:\n{pupil_outcomes}
Key learning points:\n{klps}

Previous Unit: {previous_unit}
Previous Unit Description:\n{previous_unit_desc}

Next Unit: {next_unit}
Next Unit Description:\n{next_unit_desc}

HERE IS A LIST OF PREVIOUS THREAD PROGRESSION STATEMENTS GENERATED IN ORDER:
\n{prior_progression}


OUTPUT REQUIRED:

Write a one-paragraph thread progression statement for every unitId. 
This paragraph is a detailed explanation of how the thread progresses 
within the current unit, using specific unit content. This is the most 
important part of the paragraph. Focus on what pupils learn that 
contributes to the long-term development of the thread.

- The paragraph must explain how the thread progresses within this unit.
- It should refer to specific people, events, themes, or concepts from the unit content.
- It must describe what pupils learn that adds to their cumulative understanding of the thread.
- Avoid repeating the unit description. Instead, highlight what is new, deepened, or extended in the thread's journey.

Style guide:
- Write in British English
- Write in the present tense
- Begin each paragraph with the words "In this unit, the thread progresses... "
- Do NOT repeat the thread title or description anywhere in the paragraph. 
    For example, do not say "In this unit, the thread of invasion, migration, 
    and settlement progresses...".
- Use Bloom's command language - specific, level-appropriate active verbs like: 
    creates, applies, evaluates, understands, progresses
- Always use the term 'pupils' rather than 'students'
- Write economically and precisely
- MAXIMUM PARAGRAPH LENGTH: 500 characters.


EXAMPLE:

In this unit, the thread progresses by evaluating the impact of Viking 
and Norman invasions on medieval Britain, highlighting significant 
changes in governance, culture, and identity. Pupils explore the 
creation of the Danelaw, the establishment of the feudal system, and the 
cultural connections forged between England and France through the 
Angevin Empire. By examining these events, pupils understand how 
successive waves of conquest and settlement shape societal structures 
and contribute to a distinct English identity, enriching their 
comprehension of invasion, migration, and settlement dynamics.

FINAL IMPORTANT REMINDER: 
The paragraph that you generate MUST HAVE A MAXIMUM of 500 characters.

"""

    return prompt


# ------------------------
# 4. Generate completions using LLM
# ------------------------

def generate_statements(df: pd.DataFrame, model_name: str = "gpt-4o") -> pd.DataFrame:
    unit_groups = df.groupby("unitId")
    results = []
    progression_so_far = []
    max_prev_statements = 10

    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    for unit_id, group in unit_groups:
        unit = group.iloc[0]  # Get the current row
        current_board = unit['examBoardTitle']
        
        # FILTER prior progression statements by exam board
        if current_board == "NoBoard":
            allowed_statements = [
                s["statement"] for s in progression_so_far if s["examBoardTitle"] == "NoBoard"
            ]
        else:
            allowed_statements = [
                s["statement"] for s in progression_so_far if s["examBoardTitle"] in [current_board, "NoBoard"]
            ]

        # BUILD the prior progression string (limit to max_prev_statements)
        prior_progression = "\n".join(f"{i+1}. {s}" for i, s in enumerate(allowed_statements[-max_prev_statements:]))

        # CONSTRUCT the prompt
        prompt = construct_prompt(group, prior_progression=prior_progression)
        
        print(prompt)

        # Send prompt to LLM
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            completion = response.choices[0].message.content.strip()
            
            # Save completion along with its board info
            progression_so_far.append({
                "statement": completion,
                "examBoardTitle": current_board
            })
            
        except Exception as e:
            completion = f"Error: {str(e)}"

        group = group.copy()
        group["threadProgressionStatement_para2"] = completion
        results.append(group)

    return pd.concat(results)


# ------------------------
# 5. Save to CSV
# ------------------------

def save_to_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)
    print(f"✅ Saved results to '{path}'")


# ------------------------
# 6. Main Execution (Modular)
# ------------------------

if __name__ == "__main__":
    input_csv = "history_thread_132.csv"
    output_csv = "history_thread_132_test_changes.csv"

    # Step 1: Enrich DataFrame (can inspect at this stage)
    enriched_df = load_and_enrich_dataframe(input_csv)
    enriched_df["threadProgressionStatement_para1"] = enriched_df.apply(make_paragraph1, axis=1)
    deduped_df = enriched_df.drop_duplicates(subset=["threadProgressionStatement_para1"])

    # Step 2: Uncomment below when ready to run LLM queries
    final_df = generate_statements(deduped_df)

    # Concatenate paras 1 and 2 with a blank line
    final_df["threadProgressionStatement"] = (
        final_df["threadProgressionStatement_para1"].fillna("") +
        "\n\n" +
        final_df["threadProgressionStatement_para2"].fillna("")
    )
    
    # Sort the DataFrame
    final_df = final_df.sort_values(by=['yearGroup', 'unitOrder'])

    # Add a sequential row number starting from 1
    final_df['unitThreadOrder'] = range(1, len(final_df) + 1)

    # Select desired columns
    columns_to_keep = [
        'threadTitle',
        'threadDescription',
        'keyStageTitle',
        'yearTitle',
        'examBoardTitle',
        'unitId',
        'unitTitle',
        'unitDescription',
        'unitThreadOrder',
        'threadProgressionStatement'
    ]
    final_df = final_df[columns_to_keep]
    

    # Save cleaned file
    save_to_csv(final_df, output_csv)
    
