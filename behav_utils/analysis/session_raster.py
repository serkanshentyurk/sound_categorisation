def compute_session_raster(session) -> Dict:
    """
    Extract trial-by-trial data from a pre-filtered session for raster plotting.

    Args:
        session: Pre-filtered SessionData.

    Returns:
        Dict with:
            'stimuli': array
            'choices': array
            'categories': array
            'correct': bool array
            'no_response': bool array
            'n_trials': int
            'session_id': str
            'session_idx': int
    """
    arrays = session.get_arrays()
    choices = arrays['choices']
    categories = arrays['categories']
    no_resp = arrays['no_response']

    valid = ~no_resp
    correct = np.full(len(choices), False)
    correct[valid] = (choices[valid] == categories[valid])

    return {
        'stimuli': arrays['stimuli'],
        'choices': choices,
        'categories': categories,
        'correct': correct,
        'no_response': no_resp,
        'n_trials': len(choices),
        'session_id': session.session_id,
        'session_idx': session.session_idx,
    }
