"""Minimal evaluation adapter for the existing movie_rag Resource Assistant.

Intentionally independent of src/movie_rag. Does not perform retrieval or access
private ratings. Keep read-only retrieval results distinct from answer text.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VisibleMovie:
    movie_id: int
    title: str
    rating: float


@dataclass(frozen=True)
class QueryTemplate:
    template_id: str
    renderer: Callable[[list[VisibleMovie]], str]


def title_rating_v1(movies: list[VisibleMovie]) -> str:
    """Stable, human-readable query. No tags, genres, descriptions or LLM calls."""
    rows = sorted(movies, key=lambda x: x.movie_id)
    if not rows:
        return 'Vorrei dei consigli su film da vedere. Non ho ancora indicato preferenze.'
    lines = ['Consigliami dei film che potrebbero piacermi sulla base delle valutazioni che ho fornito.',
             'Le stelle indicano quanto ho apprezzato ciascun film (scala 0,5-5):']
    for m in rows:
        lines.append(f'- [{m.movie_id}] {m.title}: {m.rating:.1f}/5 stelle')
    lines.extend(['Considera sia le valutazioni alte sia quelle basse.',
                  'Suggerisci film diversi da quelli già indicati.'])
    return '\n'.join(lines)


TEMPLATES: dict[str, QueryTemplate] = {
    'title_rating_v1': QueryTemplate('title_rating_v1', title_rating_v1),
}


def register_template(name: str, renderer: Callable[[list[VisibleMovie]], str]) -> None:
    if not name or name in TEMPLATES:
        raise ValueError('Name empty or already defined')
    TEMPLATES[name] = QueryTemplate(name, renderer)


def make_query(frame: pd.DataFrame, metadata: Mapping[int, str], template_id: str) -> str:
    if template_id not in TEMPLATES:
        raise KeyError(f'Unknown template {template_id}; available {list(TEMPLATES)}')
    movies = [VisibleMovie(int(r.movieId), str(metadata[int(r.movieId)]), float(r.rating))
              for r in frame.itertuples(index=False)]
    return TEMPLATES[template_id].renderer(movies)


def input_digest(frame: pd.DataFrame, metadata: Mapping[int, str], template_id: str) -> str:
    payload = {'template': template_id,
               'items': sorted([{'movieId': int(r.movieId), 'rating': float(r.rating),
                                 'title': str(metadata[int(r.movieId)])}
                                for r in frame.itertuples(index=False)], key=lambda m:m['movieId'])}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()


def field(item, attr, default=None):
    if isinstance(item, Mapping):
        return item.get(attr, default)
    return getattr(item, attr, default)


def extract_rankings(state, *, catalog: set[int], shown: set[int],
                     target_k: int = 10) -> dict:
    """Use the MCP retrieval result, never parse answer prose for the primary ranking.

    No popularity padding: a short list stays short (observable incomplete result).
    """
    response = field(state, 'search_response')
    matches = field(response, 'results', ()) or ()
    if not isinstance(matches, (list, tuple)):
        raise TypeError('search_response.results is not a list/tuple; adapt contract explicitly')
    raw = []
    unknown = []
    for hit in matches:
        movie_id = field(hit, 'resource_id')
        if movie_id is None:
            raise ValueError('MCP hit without resource_id: cannot evaluate')
        try:
            movie_id = int(movie_id)
        except Exception as exc:
            raise ValueError(f'Non-numeric resource_id: {movie_id!r}') from exc
        score = field(hit, 'score', None)
        rank = field(hit, 'rank', len(raw) + 1)
        if movie_id not in catalog:
            unknown.append(movie_id)
            continue
        raw.append({'movieId': movie_id, 'source_rank': int(rank),
                    'score': float(score) if score is not None else None,
                    'title': str(field(hit, 'title', ''))})
    if unknown:
        raise ValueError(f'MCP returned IDs outside MovieLens catalog: {unknown[:12]}')
    raw.sort(key=lambda r: r['source_rank'])
    candidates=[];seen=set();excluded=[]
    for item in raw:
        mid=item['movieId']
        if mid in shown:
            excluded.append(mid)
            continue
        if mid in seen:
            continue
        seen.add(mid)
        candidates.append(item)
    primary=[]
    for i,r in enumerate(candidates[:target_k],start=1):
        primary.append({'movieId':r['movieId'], 'rank':i, 'score':r['score'], 'title':r['title']})
    ans = field(state, 'final_answer')
    answer_items = field(ans, 'recommendations', ()) or ()
    cited=[];seen_answer=set();unmapped=[]
    for rec in answer_items:
        mid=field(rec, 'resource_id')
        if mid is None:
            mid=field(rec, 'movieId')
        try:mid=int(mid)
        except (ValueError,TypeError):
            unmapped.append(str(mid))
            continue
        if mid in shown or mid not in catalog:
            unmapped.append(str(mid))
            continue
        if mid not in seen_answer:
            cited.append(mid);seen_answer.add(mid)
    # No silent assumption that answer IDs necessarily equal retrieval IDs.
    status = field(state, 'status', 'unknown')
    return {'retrieval':primary, 'answer_ids':cited,
            'answer_unmapped':unmapped, 'raw_retrieved':len(raw),
            'eligible_retrieved':len(candidates), 'shown_excluded':excluded,
            'returned':len(primary), 'shortfall':target_k-len(primary),
            'state_status':str(status),
            'mcp_status':str(field(response,'status','missing')),
            'mcp_query':field(response,'query'),
            'final_answer_text':field(ans,'answer'),
            'response_missing':response is None}


def score_rankings(predictions:pd.DataFrame, *, fixed:pd.DataFrame, extra:pd.DataFrame,
                   private_profile:pd.DataFrame, positive_min=4.0, top_k=10,
                   expected_keys: pd.DataFrame | None = None) -> pd.DataFrame:
    """Evaluate A/B only *after* ranking is computed. Fewer than K results are allowed.

    Missing slots have zero gain. A: fixed 20; B: fixed + extra + unrevealed
    part of 10-item profile. This function is NOT imported by the live executor.
    """
    required={'scenario_id','k','model','movieId','rank'}
    if not required.issubset(predictions.columns):
        raise ValueError(f'Missing prediction columns: {required-set(predictions.columns)}')
    truth={}
    for name,src in [('F',fixed),('E',extra),('P',private_profile)]:
        for sid,g in src.groupby('scenario_id'):
            recs=[]
            for row in g.itertuples(index=False):
                recs.append((int(row.movieId),float(row.rating),int(row.reveal_order) if name=='P' else 0))
            truth.setdefault(str(sid),{})[name]=recs
    rows=[]
    groups=list(predictions.groupby(['scenario_id','k','model']))
    actual={tuple(x) for x in predictions[['scenario_id','k','model']].itertuples(index=False,name=None)}
    if expected_keys is not None:
        expected={tuple(x) for x in expected_keys[['scenario_id','k','model']].itertuples(index=False,name=None)}
        if not actual.issubset(expected):
            raise ValueError('Unexpected scenario/k/model keys in ranking')
        # An empty MCP retrieval is an observable result, not a missing experiment.
        # It receives zero gain at every position; other baselines retain their rows.
        for key in sorted(expected-actual):
            groups.append((key, predictions.iloc[0:0].copy()))
    for (sid,k,model),g in groups:
        sid=str(sid); k=int(k); model=str(model)
        ordered=g.sort_values('rank')
        if len(ordered)>top_k or ordered.movieId.duplicated().any() or \
                ordered['rank'].tolist()!=list(range(1,len(ordered)+1)):
            raise ValueError(f'Malformed top-k for {sid}/{k}/{model}')
        if sid not in truth:raise ValueError(f'Unknown scenario {sid}')
        f=truth[sid].get('F',[])
        e=truth[sid].get('E',[])
        p=[entry for entry in truth[sid].get('P',[]) if entry[2]>k]
        shown={entry[0] for entry in truth[sid].get('P',[]) if entry[2]<=k}
        if set(ordered.movieId)&shown:raise ValueError(f'Revealed film recommended: {sid}, k={k}')
        for scheme,records in [('A',f),('B',f+e+p)]:
            pos={movie for movie,rating,_ in records if rating>=positive_min}
            ids=ordered.movieId.astype(int).tolist()
            labels=np.asarray([mid in pos for mid in ids],dtype=float)
            npos=len(pos)
            if npos:
                discount=1/np.log2(np.arange(2,len(labels)+2))
                ideal=(1/np.log2(np.arange(2,min(top_k,npos)+2))).sum()
                ndcg=float((labels*discount).sum()/ideal)
                recall=float(labels.sum()/npos)
                hit=float(labels.sum()>0)
            else:ndcg=recall=hit=float('nan')
            rows.append({'scenario_id':sid,'k':k,'model':model,'scheme':scheme,
                         'ndcg@10':ndcg,'recall@10':recall,'hr@10':hit,
                         'n_relevant':npos,'n_returned':len(ids),'shortfall':top_k-len(ids),
                         'n_unjudged_topk':sum(mid not in {v[0] for v in records} for mid in ids)})
    out=pd.DataFrame(rows)
    return out


def macro_by_user(per_scenario:pd.DataFrame, owners:Mapping[str,int]):
    metrics=['ndcg@10','recall@10','hr@10','n_returned','shortfall','n_unjudged_topk']
    df=per_scenario.copy()
    df['userId']=df.scenario_id.map(owners)
    if df.userId.isna().any():raise ValueError('Missing user mapping')
    per_user=df.groupby(['userId','k','model','scheme'],as_index=False)[metrics].mean()
    macro=per_user.groupby(['k','model','scheme'],as_index=False)[metrics].mean()
    return per_user,macro
