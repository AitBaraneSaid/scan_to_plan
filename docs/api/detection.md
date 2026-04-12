# API — Détection

::: scan2plan.detection.line_detection
    options:
      members:
        - DetectedSegment
        - detect_lines_hough
      show_source: true

::: scan2plan.detection.openings
    options:
      members:
        - Opening
        - detect_openings_along_wall
      show_source: true

::: scan2plan.detection.curved_walls
    options:
      members:
        - DetectedArc
        - DetectedPillar
        - detect_curved_walls
        - detect_pillars
      show_source: true
