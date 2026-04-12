# API — Contrôle qualité

::: scan2plan.qa.validator
    options:
      members:
        - validate_plan
        - generate_qa_report
      show_source: true

::: scan2plan.qa.zone_scoring
    options:
      members:
        - ZoneScore
        - ZoneMap
        - compute_zone_scores
        - generate_confidence_heatmap
        - export_low_confidence_zones_to_dxf
        - generate_pdf_report
      show_source: true

::: scan2plan.qa.metrics
    options:
      members:
        - QAReport
      show_source: true
