"""Scoring only: references are never recognition inputs."""
from .ocr_visual import edit_distance


def region_text_counts(words,gold,matches):
    if len(words)!=len(matches): raise ValueError('Prediction/matching length mismatch')
    used=set(); errors=correct=matched_errors=matched_chars=0
    for word,g in zip(words,matches):
        if g>=0:
            if g>=len(gold) or g in used: raise ValueError('Invalid or duplicate GT match')
            used.add(g); distance=edit_distance(gold[g],word)
            errors+=distance; matched_errors+=distance; matched_chars+=len(gold[g]); correct+=gold[g]==word
        else: errors+=len(word)
    errors+=sum(len(text) for i,text in enumerate(gold) if i not in used)
    return {'gt_words':len(gold),'pred_words':len(words),'correct_words':correct,'matched':len(used),
            'gt_characters':sum(map(len,gold)),'character_errors':errors,'matched_characters':matched_chars,'matched_errors':matched_errors}
