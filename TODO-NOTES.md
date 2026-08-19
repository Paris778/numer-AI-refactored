# TODO 
---------------------------------------

Add numerAI MCP for agents 

Do new benchmark models (simple and trivial) 
---- Also add all the old ones (try to recreate them) - OneNote

Update libraries (lgbm etc) 

For a personal project, this is above-average and very respectable; for a team or production system, it still has too much research-code entropy and too little platform discipline. (maybe give to a powerful model to restructure etc, do better modules etc.)


Deliberately deferred — consciously ignored

   • test_analysis.py monolith and duplicated MetricScorecard builders — works fine; split only when editing that subsystem.
     Churn without functional gain.
   • Private cross-module imports (meta.py:480 et al.) — promote during the next features.py refactor, not standalone.
   • Dead placeholders (book_correlation, redundancy metrics) — E6-deferred by plan; add TODO markers if touched.
   • Duplicate _train_multi_target_oof — consolidate when one of the two copies next changes.
   • _gpu axis semantics, opt.py broad except, era_col plumbing — edge or deliberate today; revisit on first contact, not
     proactively.
   • Warning volume, O(n²) set rebuild, open_browser default — cosmetic; fold into future edits.
   • evaluation-bible v5.2 facts, canon target-name staleness, skill command nits, docs/superpowers mapping — batch into the
     next docs-hygiene pass.


    ----------

     > TODO : Augment the dashboard with Gemini , 
     > TODO : refactor and put all the front end stuff into one directory for logical isolation 
     > TODO : remove all    │
 │   the legacy stuff and don't use plotly and integrated graphs. Use html css and javascript from scratch. One       │
 │   module is only responsible for front end stuff. Self contained.   


 -----------------

 Is there any benefit to the cofig model style ? Why not python files ? 
 The current config does not allow for custom feature neutralisation , or feature engineering - Refactor potentially 