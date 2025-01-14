import pandas as pd

def call_context_filter(txt_file_path):
    """
    Filters spurious detections based on frequency, duration, and occurrence criteria,
    applied separately within each WAV file.

    Args:
        txt_file_path (str): Path to the input text file containing predictions.
    """
    # Load the text file into a Pandas DataFrame
    df = pd.read_csv(txt_file_path, sep='\t')
    
    # Ensure necessary columns are present
    required_columns = [
        'wav_file_path', 'label', 'score', 'start_time_sec', 'end_time_sec',
        'min_frequency', 'max_frequency'
    ]
    if not all(col in df.columns for col in required_columns):
        raise ValueError("Input file missing required columns.")
    
    # Filter by score threshold
    df = df[df['score'] >= 0.1]

    # Define helper functions for filtering
    def join_close_calls(group, join_within_sec):
        if group.empty:
            return group
        group = group.sort_values(by='start_time_sec')
        merged_calls = []
        current_call = None

        for _, row in group.iterrows():
            if current_call is None:
            # Start a new merge group with the current call
                current_call = row.to_dict()
            elif row['start_time_sec'] - current_call['end_time_sec'] <= join_within_sec:
                    # Extend the current call
                current_call['end_time_sec'] = max(current_call['end_time_sec'], row['end_time_sec'])
                current_call['score'] = max(current_call['score'], row['score'])
            else:
                    # Current call is too far from the previous, save the current and start a new one
                merged_calls.append(current_call)
                current_call = row.to_dict()

    # Append the last merged call if not None
        if current_call is not None:
            merged_calls.append(current_call)

        return pd.DataFrame(merged_calls)

    def keep_highest_scoring_within_time(group, within_sec):
        if group.empty:
            return group
        if 'start_time_sec' not in group.columns or 'score' not in group.columns:
            print("Missing columns in group:", group.columns)
            return group  # Return the group as is if columns are missing
        group = group.sort_values(by=['start_time_sec', 'score'], ascending=[True, False])
        kept_calls = []
        prev_end_time = -float('inf')
        for _, row in group.iterrows():
            if row['start_time_sec'] - prev_end_time > within_sec:
                kept_calls.append(row)
                prev_end_time = row['start_time_sec']
        return pd.DataFrame(kept_calls)

    def filter_call_type(group, label, freq_range, duration_range, join_within_sec=None, keep_highest_within_sec=None):
        filtered_group = group[(group['label'] == label) & 
                               (group['min_frequency'] >= freq_range[0]) & 
                               (group['max_frequency'] <= freq_range[1])]

        # Handle duration filtering
        if duration_range:
            durations = filtered_group['end_time_sec'] - filtered_group['start_time_sec']
            filtered_group = filtered_group[(durations >= duration_range[0]) & (durations <= duration_range[1])]

        # Join calls within a specific time frame
        if join_within_sec is not None:
            filtered_group = join_close_calls(filtered_group, join_within_sec)

        # Keep the highest scoring call within a specific time frame
        if keep_highest_within_sec is not None:
            filtered_group = keep_highest_scoring_within_time(filtered_group, keep_highest_within_sec)

        return filtered_group

    def apply_wav_filters(group):
        if group.empty:
            return group
        # Apply the individual WAV file filters
        label_counts = group['label'].value_counts()
        if label_counts.get('A', 0) < 3 and label_counts.get('B', 0) < 3:
            group = group[~group['label'].isin(['A', 'B'])]
        if label_counts.get('D', 0) < 1:
            group = group[group['label'] != 'D']
        # Filter for `40Hz` calls
        if (label_counts.get('40Hz', 0) < 3 or label_counts.get('20Hz', 0) < 3) and not (
                label_counts.get('20Hz', 0) > 5 and label_counts.get('40Hz', 0) >= 1):
            group = group[group['label'] != '40Hz']

        # Filter for `20Hz` calls
        if label_counts.get('20Hz', 0) < 5:
            group = group[group['label'] != '20Hz']
        return group

    # Define filter parameters
    filters = {
        'A': {'freq_range': (70, 90), 'duration_range': (3, 20), 'join_within_sec': 5},
        'B': {'freq_range': (10, 70), 'duration_range': (3, 20), 'join_within_sec': 5},
        'D': {'freq_range': (20, 80), 'duration_range': (2, 8)},
        '40Hz': {'freq_range': (35, 100), 'duration_range': (0, 3)},
        '20Hz': {'freq_range': (9, 35), 'duration_range': None, 'join_within_sec': 2, 'keep_highest_within_sec': 2},
    }

    # Apply filters for each WAV file separately
    def process_wav_group(group):
        if group.empty:
            return group
        # Apply call type filters
        filtered_groups = []
        for label, params in filters.items():
            filtered_groups.append(filter_call_type(group, label, **params))
        filtered_group = pd.concat(filtered_groups)

        # Apply individual WAV file filters
        return apply_wav_filters(filtered_group)

    filtered_df = df.groupby('wav_file_path', group_keys=False).apply(process_wav_group)

    # Save the filtered predictions to a new file
    output_file_path = txt_file_path.replace('_raw_detections', '_context_filtered')
    filtered_df.to_csv(output_file_path, sep='\t', index=False)
    print(f"Filtered predictions saved to {output_file_path}")