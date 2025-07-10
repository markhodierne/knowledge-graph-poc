
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


def construct_prompt(row_group) -> str:
    unit = row_group.iloc[0]

    thread_title = unit['threadTitle']
    thread_desc = unit['threadDescription']

    previous_unit = format_unit(unit['prevUnitId'], unit['prevUnitTitle'], unit['prevUnitYear'])
    previous_unit_desc = unit['prevUnitDesc']
    
    current_unit = format_unit(unit['unitId'], unit['unitTitle'], unit['yearTitle'])
    current_unit_desc = unit['unitDescription']
    
    next_unit = format_unit(unit['nextUnitId'], unit['nextUnitTitle'], unit['nextUnitYear'])
    next_unit_desc = unit['nextUnitDesc']

    pupil_outcomes = unit['pupilLessonOutcome']
    klps = unit['keyLearningPoints']
    
    if pd.isna(unit['prevUnitId']) or pd.isna(unit['prevUnitTitle']) or pd.isna(unit['prevUnitYear']):
        start_text = "In this unit, the thread is established... "
        prompt_task = f"""
            Write a one-paragraph that introduces this first unit in the thread sequence. 
            This paragraph is a detailed explanation of how the thread begins, 
            using specific unit content. Focus on what pupils learn that 
            contributes to the long-term development of the thread.

            - The paragraph must explain how the thread progresses within this unit.
            - It should refer to specific people, events, themes, or concepts from the unit content.
            - It must describe what pupils learn that begins their understanding of the thread.
            - Avoid repeating the unit description. Instead, highlight what is being explored in the thread's journey.
        """
    elif pd.isna(unit['nextUnitId']) or pd.isna(unit['nextUnitTitle']) or pd.isna(unit['nextUnitYear']):
        start_text = "In this unit, the thread concludes... "
        prompt_task = f"""
            Write a one-paragraph thread progression statement for the current unit. 
            This paragraph is a detailed explanation of how the thread progresses 
            within the current unit, using specific unit content. Focus on what pupils learn that 
            contributes to the long-term development of the thread.

            - The paragraph must explain how the thread progresses within this unit.
            - It should refer to specific people, events, themes, or concepts from the unit content.
            - It must describe what pupils learn that adds to their cumulative understanding of the thread.
            - Avoid repeating the unit description. Instead, highlight what is new, deepened, or extended in the thread's journey.
        """
    else:
        start_text = "In this unit, the thread progresses... "
        prompt_task = f"""
            Write a one-paragraph thread progression statement for the current unit. 
            This paragraph is a detailed explanation of how the thread progresses 
            within the current unit, using specific unit content. Focus on what pupils learn that 
            contributes to the long-term development of the thread.

            - The paragraph must explain how the thread progresses within this unit.
            - It should refer to specific people, events, themes, or concepts from the unit content.
            - It must describe what pupils learn that adds to their cumulative understanding of the thread.
            - Avoid repeating the unit description. Instead, highlight what is new, deepened, or extended in the thread's journey.
        """

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


INPUT PROVIDED:

Thread: {thread_title}
Thread description:\n{thread_desc}

Current Unit: {current_unit}
Current Unit Description:\n{current_unit_desc}

Pupil lesson outcomes:\n{pupil_outcomes}
Key learning points:\n{klps}


IMPORTANT: ONLY use the inputs provided in this prompt. DO NOT make reference to any other information in your response. 


OUTPUT REQUIRED:

{prompt_task}

STYLE GUIDE:

- Begin the paragraph with the words {start_text}
- Do NOT repeat the thread title or description anywhere in the paragraph. 
- Always use the term 'pupils' rather than 'students'
- Do not restate what is taught. Instead, explain how this extends or builds 
on what pupils have already learned.
- AVOID generalities like "pupils learn about change over time". 
Instead, use concrete references to historical people, ideas, or events.
- AVOID a bland, general final sentence that makes reference to the thread. 
Instead, keep the content specific and focused to the unit and what a pupil learns. 
- DO NOT use phrases like "deepening their understanding", "enriching their grasp", 
'deepens their grasp", "enhancing their understanding" in the final sentence.
STAY FOCUSED on what the pupil learns and be specific about historical people, 
ideas, or events.


- Write in British English
- Write in the present tense when describing the pupil's learning journey
- Use the most appropriate tense when referring to the actual content of the thread
- Use Bloom's command language - specific, level-appropriate active verbs like: 
    creates, applies, evaluates, understands, progresses
- Write economically, precisely and use language appropriate for describing the 
thread progression for a specific year group. For example, a Year 1 unit will 
be simpler in content than a Year 3 unit and the language should reflect that fact.

- MAXIMUM PARAGRAPH LENGTH: 500 characters.


EXAMPLE OUTPUTS:

Examples of excellent output include the following:

"In this unit, the thread progresses as pupils explore how Roman identity 
evolved alongside imperial expansion. They build on earlier learning about 
conquest by examining how the shift from Republic to Empire influenced power 
and control. Events like the Punic Wars and Caesar's dictatorship show how 
empires adapt to challenges."

"In this unit, the thread progresses by offering pupils further examples 
of the processes that make up imperial expansion, allowing them to compare 
and contrast the Mughal Empire's expansion and the role of violence and 
diplomacy in its conquests with earlier knowledge of Alexander, Rome and 
Europeans in the Americas. Pupils assess figures like Babur and Akbar, 
exploring how military victories and policies such as religious tolerance 
shaped the Mughal Empire."

"In this unit, the thread progresses by exploring Roman imperial control and 
resistance in Britain. Pupils analyse Claudius's invasion, Boudica's rebellion, 
and Roman cultural imposition through architecture and religion. They 
understand how Roman conquest reshaped British life, with some adopting 
Roman ways while others resist, thereby deepening their grasp of empire by 
providing a new focus on the cultural interaction between colonisers and the 
colonised."


FINAL IMPORTANT REMINDER: 
The paragraph that you generate MUST HAVE A MAXIMUM of 500 characters.
Check your output to ensure that it complies with the STYLE GUIDE above.

"""

    return prompt


# ------------------------
# 4. Generate completions using LLM
# ------------------------

def generate_statements(df: pd.DataFrame, model_name: str = "gpt-4o") -> pd.DataFrame:
    unit_groups = df.groupby("unitId")
    results = []

    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    for unit_id, group in unit_groups:
        unit = group.iloc[0]  # Get the current row

        # CONSTRUCT the prompt
        prompt = construct_prompt(group)

        # Send prompt to LLM
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            completion = response.choices[0].message.content.strip()
            
        except Exception as e:
            completion = f"Error: {str(e)}"

        group = group.copy()
        group["threadProgressionStatement_para2"] = completion
        results.append(group)

    return pd.concat(results)


# Function not currently used - not performing well
def identify_relevant_prior_topics(df: pd.DataFrame, model_name: str = "gpt-4o") -> pd.DataFrame:
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    enriched_groups = []

    for thread, group in df.groupby("threadTitle"):
        group = group.sort_values(by="unitThreadOrder").copy()
        group["priorTopicsSummary"] = ""

        for i, row in group.iterrows():
            current_text = row["threadProgressionStatement_para2"]

            # Gather all earlier para2 statements in this thread
            prior_units = group[group["unitThreadOrder"] < row["unitThreadOrder"]]
            prior_paragraphs = prior_units["threadProgressionStatement_para2"].dropna().tolist()

            if not prior_paragraphs:
                group.at[i, "priorTopicsSummary"] = ""
                continue

            prior_context = "\n".join([f"- {p}" for p in prior_paragraphs])

            prompt = f"""
You are analysing a sequence of history curriculum paragraphs for a single thread.

The CURRENT unit's paragraph is:
"{current_text}"

Here are the thread progression paragraphs from EARLIER units in this thread:
{prior_context}

🎯 TASK:
Identify up to THREE specific historical people, ideas, or events from the EARLIER units that the CURRENT unit builds on.

ONLY include items that:
- Are clearly introduced in prior units
- Are directly relevant to what pupils are learning in the current unit
- Represent **specific historical content**, not general skills or concepts

If the current unit introduces new ideas unrelated to earlier units, return an empty string.

🚫 Do NOT include general phrases like:
- "societal changes"
- "cultural fusion"
- "conflict"
- "migration patterns"
- "continuity and change"

✅ Valid examples:
- "British Nationality Act 1948", "Norman Conquest", 
"Battle of Hastings", "Windrush generation", "Edward the Confessor"

FORMAT:
Return your result as one line starting with:
Building on previous topics: topic1, topic2, topic3

If no prior topics are relevant, return a blank line.

"""

            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                topic_line = response.choices[0].message.content.strip()

                # Enforce format: clean up if LLM added extra text
                if topic_line.lower().startswith("building on previous topics"):
                    summary = topic_line
                else:
                    summary = f"Building on previous topics: {topic_line}" if topic_line else ""

            except Exception as e:
                summary = f"Error extracting topics: {str(e)}"

            group.at[i, "priorTopicsSummary"] = summary

        enriched_groups.append(group)

    return pd.concat(enriched_groups)


# Function not currently used - not performing well
def refine_statements_with_prior_units(df: pd.DataFrame, model_name: str = "gpt-4o") -> pd.DataFrame:
    from openai import OpenAI
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    refined_groups = []

    for thread, group in df.groupby("threadTitle"):
        group = group.sort_values(by="unitThreadOrder").copy()
        group["threadProgressionStatement_enhanced"] = ""
        group["enhancementPrompt"] = ""  # NEW: store the actual prompt used

        for i, row in group.iterrows():
            current_text = row["threadProgressionStatement_para2"]

            # Gather all prior units' para2
            prior_units = group[group["unitThreadOrder"] < row["unitThreadOrder"]]
            prior_paragraphs = prior_units["threadProgressionStatement_para2"].dropna().tolist()

            # Skip enhancement if no prior units exist
            if not prior_paragraphs:
                group.at[i, "threadProgressionStatement_enhanced"] = current_text
                group.at[i, "enhancementPrompt"] = "No prior units — no enhancement prompt used."
                continue

            # Construct the enhancement prompt
            prior_context = "\n".join([f"- {p}" for p in prior_paragraphs])
            prompt = f"""
Below is a thread progression paragraph for a unit in a history curriculum:

"{current_text}"

And here is a list of earlier thread progression paragraphs from previous units in the same thread:

{prior_context}

✳️ TASK:
Rewrite the current paragraph, blending in any **relevant and meaningful** content 
from earlier units **only if it strengthens the coherence** of the paragraph.

Do NOT add content unless there is a **direct, content-specific connection**. 
If no clear connection exists, simply rewrite the paragraph for fluency and flow — do not force a reference.

✳️ STYLE GUIDE:
- Rewrite the paragraph fully — do NOT just append sentences.
- Paragraph MUST remain under 650 characters in length.
- If you include prior content, it must reference specific historical people, events, or concepts.
- Do not say things like "as seen previously" or "as mentioned before."
- Do not name unit titles. Use historical content directly.
- Write in British English, and describe the learning journey of the pupil in the present tense.
- Use precise, economical language with Bloom-style verbs.
- VERY IMPORTANT: Preserve the original purpose and detail of the paragraph: explaining how the thread develops in this unit.

"""

            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5
                )
                enhanced = response.choices[0].message.content.strip()
            except Exception as e:
                enhanced = f"Error enhancing: {str(e)}"

            group.at[i, "threadProgressionStatement_enhanced"] = enhanced
            group.at[i, "enhancementPrompt"] = prompt.strip()  # Store prompt

        refined_groups.append(group)

    return pd.concat(refined_groups)


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
    input_csv = "history_thread_135.csv"
    output_csv = "history_thread_135_results_v2.csv"

    # Enrich DataFrame (can inspect at this stage)
    enriched_df = load_and_enrich_dataframe(input_csv)
    enriched_df["threadProgressionStatement_para1"] = enriched_df.apply(make_paragraph1, axis=1)
    deduped_df = enriched_df.drop_duplicates(subset=["threadProgressionStatement_para1"])

    # Uncomment below when ready to run LLM queries
    final_df = generate_statements(deduped_df)
    
    # Sort the dataframe
    final_df = final_df.sort_values(by=['yearGroup', 'unitOrder'])
    final_df['unitThreadOrder'] = range(1, len(final_df) + 1)
    
    # Refine using prior units' statements - not currently working well
    #final_df = identify_relevant_prior_topics(final_df)

    # Concatenate paras 1 and 2 with a blank line
    final_df["threadProgressionStatement"] = (
        final_df["threadProgressionStatement_para1"].fillna("") +
        "\n\n" +
        final_df["threadProgressionStatement_para2"].fillna("")
    )

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
        'threadProgressionStatement',
        #'priorTopicsSummary'
    ]
    final_df = final_df[columns_to_keep]
    

    # Save cleaned file
    save_to_csv(final_df, output_csv)
    
