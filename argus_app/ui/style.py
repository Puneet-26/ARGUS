"""
ARGUS Design System - Modern Developer Forensics Theme
Case: PIXEL PURSUIT · ROHAN MEHTA

Strict color tokens, restrained dark surfaces, and teal interactive accents.
No emoji navigation, no neon gradients, no decorative charts.
"""

# Exact color tokens specified in brief
COLORS = {
    # Ported from the website casefile theme
    'app_bg': '#141618',
    'sidebar': '#141618',
    'surface': '#1A1C1E',
    'elevated': '#222528',
    'border': '#2C2F33',

    # Accents
    'primary': '#C45E51',
    'primary_hover': '#D06B5E',
    'primary_pressed': '#A94F44',

    # Text
    'text_main': '#E2DED5',
    'text_secondary': '#A8ABA5',
    'text_muted': '#9A9D98',
    'amber': '#D0A15C',

    # Forensics & Status
    'alert': '#C45E51',
    'success': '#78C6A1',          # Verified results & completed milestones

    'canvas_bg': '#101213',

    # Backward compatibility aliases for existing widgets
    'bg_darkest': '#141618',
    'bg_dark': '#141618',
    'bg_panel': '#1A1C1E',
    'bg_medium': '#222528',
    'bg_light': '#222528',
    'accent_teal': '#D0A15C',
    'accent_cyan': '#D0A15C',
    'accent_blue': '#C45E51',
    'text_primary': '#E2DED5',
    'text_dim': '#9A9D98',
    'text_mono': '#E2DED5',
    'border_subtle': '#2C2F33',
    'hover': '#222528',
    'active': '#2C2F33',
    'error': '#C45E51',
    'warning': '#D0A15C',
    'info': '#8B8E89',
}

FONTS = {
    # Native Windows fonts provide a local, network-free fallback for the web theme.
    'family_ui': '"Segoe UI"',
    'family_mono': '"Consolas"',
    'size_xs': '11px',
    'size_sm': '12px',
    'size_base': '14px',
    'size_md': '15px',
    'size_lg': '17px',
    'size_xl': '20px',
    'size_2xl': '24px',
}

SPACING = {
    'xs': '4px',
    'sm': '8px',
    'md': '12px',
    'lg': '16px',
    'xl': '20px',
    '2xl': '28px',
}

RADIUS = {
    'sm': '3px',
    'md': '5px',
    'lg': '8px',
}

MAIN_STYLE = f"""
QMainWindow {{
    background-color: {COLORS['app_bg']};
    color: {COLORS['text_main']};
}}

QWidget {{
    background-color: {COLORS['app_bg']};
    color: {COLORS['text_main']};
    font-size: {FONTS['size_base']};
    font-family: {FONTS['family_ui']};
}}

/* Casefile navigation */
QFrame#topBar {{
    background-color: {COLORS['surface']};
    border-bottom: 1px solid {COLORS['border']};
}}

QPushButton#caseNavButton {{
    height: 48px;
    padding: 0 10px;
    color: {COLORS['text_muted']};
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    border-radius: 0;
    font-size: 14px;
    font-weight: 500;
}}

QPushButton#caseNavButton:hover {{
    color: {COLORS['text_main']};
    background: transparent;
}}

QPushButton#caseNavButton:checked {{
    color: {COLORS['text_main']};
    border-bottom: 2px solid {COLORS['primary']};
    font-weight: 600;
}}

QLabel#brandName {{ font-size: 14px; font-weight: 700; letter-spacing: 1px; }}
QLabel#brandMeta {{ color: {COLORS['text_muted']}; font-size: 10px; font-family: {FONTS['family_mono']}; }}
QLabel#systemStatus {{
    color: {COLORS['amber']};
    background: {COLORS['elevated']};
    border: 1px solid {COLORS['border']};
    border-radius: 3px;
    padding: 4px 8px;
    font-family: {FONTS['family_mono']};
    font-size: 10px;
}}

/* Buttons */
QPushButton {{
    background-color: {COLORS['elevated']};
    color: {COLORS['text_main']};
    border: 1px solid {COLORS['border']};
    padding: 6px 14px;
    border-radius: {RADIUS['md']};
    font-size: {FONTS['size_base']};
    font-weight: 500;
}}

QPushButton:hover {{
    background-color: {COLORS['surface']};
    border-color: {COLORS['primary']};
    color: {COLORS['text_main']};
}}

QPushButton:pressed {{
    background-color: {COLORS['elevated']};
}}

QPushButton:disabled {{
    background-color: {COLORS['surface']};
    color: {COLORS['text_muted']};
    border-color: {COLORS['border']};
}}

QPushButton#primaryButton {{
    background-color: {COLORS['primary']};
    color: #111416;
    border: 1px solid {COLORS['primary']};
    font-weight: 600;
}}

QPushButton#primaryButton:hover {{
    background-color: {COLORS['primary_hover']};
    border-color: {COLORS['primary_hover']};
}}

QPushButton#primaryButton:pressed {{
    background-color: {COLORS['primary_pressed']};
}}

QPushButton#alertButton {{
    background-color: {COLORS['alert']};
    color: #ffffff;
    border: 1px solid {COLORS['alert']};
    font-weight: 600;
}}

/* ComboBox */
QComboBox {{
    background-color: {COLORS['elevated']};
    color: {COLORS['text_main']};
    border: 1px solid {COLORS['border']};
    padding: 5px 10px;
    border-radius: {RADIUS['md']};
    min-width: 130px;
}}

QComboBox:hover {{
    border-color: {COLORS['primary']};
}}

QComboBox::drop-down {{
    border: none;
    padding-right: 8px;
}}

QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {COLORS['text_secondary']};
    margin-right: 4px;
}}

QComboBox QAbstractItemView {{
    background-color: {COLORS['surface']};
    color: {COLORS['text_main']};
    selection-background-color: {COLORS['elevated']};
    selection-color: {COLORS['primary']};
    border: 1px solid {COLORS['border']};
    outline: none;
}}

/* Sliders */
QSlider::groove:horizontal {{
    background: {COLORS['elevated']};
    height: 4px;
    border-radius: 2px;
}}

QSlider::handle:horizontal {{
    background: {COLORS['primary']};
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
    border: 1px solid {COLORS['app_bg']};
}}

QSlider::handle:horizontal:hover {{
    background: {COLORS['primary_hover']};
}}

/* SpinBox */
QSpinBox, QDoubleSpinBox, QLineEdit {{
    background-color: {COLORS['elevated']};
    color: {COLORS['text_main']};
    border: 1px solid {COLORS['border']};
    padding: 5px 8px;
    border-radius: {RADIUS['md']};
    font-family: {FONTS['family_mono']};
    font-size: {FONTS['size_base']};
}}

QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{
    border-color: {COLORS['primary']};
}}

/* Labels */
QLabel {{
    color: {COLORS['text_main']};
    background-color: transparent;
}}

QLabel#sectionTitle {{
    font-size: {FONTS['size_md']};
    font-weight: 600;
    color: {COLORS['text_main']};
    letter-spacing: 0.2px;
}}

QLabel#secondaryLabel {{
    color: {COLORS['text_secondary']};
    font-size: {FONTS['size_sm']};
}}

QLabel#mutedLabel {{
    color: {COLORS['text_muted']};
    font-size: {FONTS['size_xs']};
    font-family: {FONTS['family_mono']};
}}

QLabel#pageTitle {{
    color: {COLORS['text_main']};
    font-size: 28px;
    font-weight: 700;
}}

QLabel#pageSummary {{
    color: {COLORS['text_secondary']};
    font-size: {FONTS['size_base']};
}}

QLabel#eyebrowLabel {{
    color: {COLORS['amber']};
    font-family: {FONTS['family_mono']};
    font-size: 12px;
    letter-spacing: 1px;
}}

QLabel#panelTitle, QLabel#sectionHeading {{
    color: {COLORS['text_main']};
    font-weight: 600;
    font-size: {FONTS['size_lg']};
}}

QLabel#statusLine {{
    color: {COLORS['text_secondary']};
    background: {COLORS['surface']};
    font-size: {FONTS['size_sm']};
}}

QLabel#infoCard, QLabel#statusCard {{
    color: {COLORS['text_secondary']};
    background: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: {RADIUS['md']};
    padding: 10px;
}}

QFrame#caseHero {{
    background: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: {RADIUS['md']};
}}

QFrame#progressCard, QFrame#summaryCard, QFrame#evidencePanel, QFrame#controlPanel {{
    background: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: {RADIUS['md']};
}}

QLabel#stageStatus {{
    color: {COLORS['text_muted']};
    background: {COLORS['elevated']};
    border-radius: 3px;
    padding: 3px 7px;
    font-size: 10px;
}}

QPushButton#secondaryButton {{
    background: transparent;
    color: {COLORS['primary']};
    border-color: {COLORS['border']};
}}

QPushButton#dangerButton {{
    background: {COLORS['surface']};
    color: {COLORS['alert']};
    border: 1px solid {COLORS['border']};
}}

QPushButton#dangerButton:hover {{
    background: {COLORS['elevated']};
    border-color: {COLORS['alert']};
}}

/* GroupBox / Forensic Card */
QGroupBox {{
    background-color: {COLORS['surface']};
    color: {COLORS['text_main']};
    border: 1px solid {COLORS['border']};
    border-radius: {RADIUS['md']};
    margin-top: 14px;
    padding: 12px 10px 10px 10px;
    font-weight: 600;
    font-size: {FONTS['size_base']};
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
    background-color: {COLORS['surface']};
    color: {COLORS['primary']};
}}

/* ScrollBars */
QScrollBar:vertical {{
    background: {COLORS['surface']};
    width: 8px;
    border-radius: 4px;
}}

QScrollBar::handle:vertical {{
    background: {COLORS['elevated']};
    border-radius: 4px;
    min-height: 24px;
}}

QScrollBar::handle:vertical:hover {{
    background: {COLORS['border']};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

QScrollBar:horizontal {{
    background: {COLORS['surface']};
    height: 8px;
    border-radius: 4px;
}}

QScrollBar::handle:horizontal {{
    background: {COLORS['elevated']};
    border-radius: 4px;
    min-width: 24px;
}}

QScrollBar::handle:horizontal:hover {{
    background: {COLORS['border']};
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
}}

/* Splitters */
QSplitter::handle {{
    background-color: {COLORS['border']};
}}

QSplitter::handle:horizontal {{
    width: 2px;
}}

QSplitter::handle:vertical {{
    height: 2px;
}}

/* CheckBox */
QCheckBox {{
    color: {COLORS['text_main']};
    spacing: 6px;
    background-color: transparent;
}}

QCheckBox:hover {{
    color: {COLORS['text_main']};
}}

QCheckBox:focus {{
    background-color: {COLORS['surface']};
}}

QCheckBox:disabled {{
    color: {COLORS['text_muted']};
}}

QTextEdit:focus, QComboBox:focus {{
    border-color: {COLORS['primary']};
}}

/* Table Widget */
QTableWidget {{
    background-color: {COLORS['surface']};
    color: {COLORS['text_main']};
    border: 1px solid {COLORS['border']};
    gridline-color: {COLORS['border']};
    font-size: {FONTS['size_sm']};
}}

QHeaderView::section {{
    background-color: {COLORS['elevated']};
    color: {COLORS['text_secondary']};
    padding: 6px;
    border: none;
    border-right: 1px solid {COLORS['border']};
    border-bottom: 1px solid {COLORS['border']};
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

/* Tab Widget */
QTabWidget::pane {{
    border: 1px solid {COLORS['border']};
    background-color: {COLORS['surface']};
    border-radius: {RADIUS['md']};
}}

QTabBar::tab {{
    background-color: {COLORS['elevated']};
    color: {COLORS['text_secondary']};
    padding: 8px 16px;
    margin-right: 2px;
    border-top-left-radius: {RADIUS['sm']};
    border-top-right-radius: {RADIUS['sm']};
    border: 1px solid {COLORS['border']};
    border-bottom: none;
}}

QTabBar::tab:selected {{
    background-color: {COLORS['surface']};
    color: {COLORS['primary']};
    font-weight: 600;
    border-bottom: 2px solid {COLORS['primary']};
}}

QTabBar::tab:hover:!selected {{
    background-color: {COLORS['surface']};
    color: {COLORS['text_main']};
}}
"""
