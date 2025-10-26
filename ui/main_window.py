"""
Main Window UI for LLM Evaluator
"""
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                                QTextEdit, QPushButton, QLabel, QLineEdit, 
                                QTableWidget, QTableWidgetItem, QFileDialog,
                                QComboBox, QSpinBox, QDoubleSpinBox, QGroupBox,
                                QTabWidget, QMessageBox, QProgressBar)
from PySide6.QtCore import Qt, QThread, Signal
import yaml
import os
from pathlib import Path
from core.model_manager import ModelManager
from core.evaluator import Evaluator
from core.config_manager import ConfigManager

class InferenceWorker(QThread):
    """Worker thread for model inference"""
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, model_manager, messages, model_name):
        super().__init__()
        self.model_manager = model_manager
        self.messages = messages
        self.model_name = model_name

    def run(self):
        try:
            result = self.model_manager.generate(self.messages, self.model_name)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Purpose-Driven Local LLM Evaluator")
        self.setGeometry(100, 100, 1400, 900)

        # Initialize managers
        self.config_manager = ConfigManager()
        self.model_manager = ModelManager(self.config_manager)
        self.evaluator = Evaluator(self.model_manager)

        # Worker thread
        self.worker = None

        self.init_ui()
        self.load_saved_state()

    def init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        # Create tab widget
        tabs = QTabWidget()

        # Task Definition Tab
        task_tab = self.create_task_tab()
        tabs.addTab(task_tab, "Task & Models")

        # Prompt & Run Tab
        prompt_tab = self.create_prompt_tab()
        tabs.addTab(prompt_tab, "Prompt & Run")

        # Evaluation Tab
        eval_tab = self.create_eval_tab()
        tabs.addTab(eval_tab, "Evaluations")

        # Settings Tab
        settings_tab = self.create_settings_tab()
        tabs.addTab(settings_tab, "Settings")

        main_layout.addWidget(tabs)

        # Status bar
        self.statusBar().showMessage("Ready")

    def create_task_tab(self):
        """Create task definition and model discovery tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Task Definition Group
        task_group = QGroupBox("Task Definition")
        task_layout = QVBoxLayout()

        task_layout.addWidget(QLabel("Describe your task:"))
        self.task_input = QTextEdit()
        self.task_input.setPlaceholderText("e.g., 'summarize legal documents', 'classify sentiment', 'creative writing'")
        self.task_input.setMaximumHeight(100)
        task_layout.addWidget(self.task_input)

        category_layout = QHBoxLayout()
        category_layout.addWidget(QLabel("Task Category:"))
        self.task_category = QComboBox()
        self.task_category.addItems(["auto-detect", "summarization", "classification", 
                                      "creative", "reasoning", "general"])
        category_layout.addWidget(self.task_category)
        category_layout.addStretch()
        task_layout.addLayout(category_layout)

        discover_btn = QPushButton("Discover Matching Models")
        discover_btn.clicked.connect(self.discover_models)
        task_layout.addWidget(discover_btn)

        task_group.setLayout(task_layout)
        layout.addWidget(task_group)

        # Model Registry Group
        registry_group = QGroupBox("Model Registry")
        registry_layout = QVBoxLayout()

        self.registry_table = QTableWidget()
        self.registry_table.setColumnCount(4)
        self.registry_table.setHorizontalHeaderLabels(["Model Name", "Type", "Tags", "Path"])
        self.registry_table.horizontalHeader().setStretchLastSection(True)
        registry_layout.addWidget(self.registry_table)

        btn_layout = QHBoxLayout()
        refresh_btn = QPushButton("Refresh Registry")
        refresh_btn.clicked.connect(self.load_registry)
        btn_layout.addWidget(refresh_btn)

        add_model_btn = QPushButton("Add Model")
        add_model_btn.clicked.connect(self.add_model_to_registry)
        btn_layout.addWidget(add_model_btn)
        btn_layout.addStretch()

        registry_layout.addLayout(btn_layout)
        registry_group.setLayout(registry_layout)
        layout.addWidget(registry_group)

        # Matched Models Group
        matched_group = QGroupBox("Matched Models for Task")
        matched_layout = QVBoxLayout()

        self.matched_list = QTextEdit()
        self.matched_list.setReadOnly(True)
        self.matched_list.setMaximumHeight(100)
        matched_layout.addWidget(self.matched_list)

        matched_group.setLayout(matched_layout)
        layout.addWidget(matched_group)

        self.load_registry()
        return widget

    def create_prompt_tab(self):
        """Create prompt editing and execution tab"""
        widget = QWidget()
        layout = QHBoxLayout(widget)

        # Left side - Prompt Editor
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        left_layout.addWidget(QLabel("System Prompt:"))
        self.system_prompt = QTextEdit()
        self.system_prompt.setPlaceholderText("Enter system prompt here...")
        self.system_prompt.setMaximumHeight(120)
        left_layout.addWidget(self.system_prompt)

        left_layout.addWidget(QLabel("User Prompt:"))
        self.user_prompt = QTextEdit()
        self.user_prompt.setPlaceholderText("Enter user prompt here...")
        left_layout.addWidget(self.user_prompt)

        btn_layout = QHBoxLayout()

        self.run_single_btn = QPushButton("Run on Selected Model")
        self.run_single_btn.clicked.connect(self.run_single_model)
        btn_layout.addWidget(self.run_single_btn)

        self.run_all_btn = QPushButton("Run on All Matched Models")
        self.run_all_btn.clicked.connect(self.run_all_models)
        btn_layout.addWidget(self.run_all_btn)

        left_layout.addLayout(btn_layout)

        prompt_file_layout = QHBoxLayout()
        save_prompt_btn = QPushButton("Save Prompt")
        save_prompt_btn.clicked.connect(self.save_prompt)
        prompt_file_layout.addWidget(save_prompt_btn)

        load_prompt_btn = QPushButton("Load Prompt")
        load_prompt_btn.clicked.connect(self.load_prompt)
        prompt_file_layout.addWidget(load_prompt_btn)
        prompt_file_layout.addStretch()

        left_layout.addLayout(prompt_file_layout)

        layout.addWidget(left_panel)

        # Right side - Response Viewer
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        right_layout.addWidget(QLabel("Model Response:"))
        self.response_viewer = QTextEdit()
        self.response_viewer.setReadOnly(True)
        right_layout.addWidget(self.response_viewer)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        right_layout.addWidget(self.progress_bar)

        # Stats
        self.stats_label = QLabel("Stats: -")
        right_layout.addWidget(self.stats_label)

        layout.addWidget(right_panel)

        return widget

    def create_eval_tab(self):
        """Create evaluation tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Eval file selection
        eval_file_layout = QHBoxLayout()
        eval_file_layout.addWidget(QLabel("Eval Set:"))
        self.eval_file_input = QLineEdit()
        eval_file_layout.addWidget(self.eval_file_input)

        browse_eval_btn = QPushButton("Browse")
        browse_eval_btn.clicked.connect(self.browse_eval_file)
        eval_file_layout.addWidget(browse_eval_btn)

        run_eval_btn = QPushButton("Run Evaluation")
        run_eval_btn.clicked.connect(self.run_evaluation)
        eval_file_layout.addWidget(run_eval_btn)

        layout.addLayout(eval_file_layout)

        # Comparison Table
        layout.addWidget(QLabel("Multi-Model Comparison:"))
        self.comparison_table = QTableWidget()
        self.comparison_table.setColumnCount(7)
        self.comparison_table.setHorizontalHeaderLabels([
            "Model Name", "Output Snippet", "Latency (s)", "Tokens", "Pass/Fail", "Rating", "Actions"
        ])
        self.comparison_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.comparison_table)

        # Export button
        export_layout = QHBoxLayout()
        export_btn = QPushButton("Export Results to CSV")
        export_btn.clicked.connect(self.export_results)
        export_layout.addWidget(export_btn)
        export_layout.addStretch()

        layout.addLayout(export_layout)

        return widget

    def create_settings_tab(self):
        """Create settings tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Model Settings Group
        model_group = QGroupBox("Model Settings")
        model_layout = QVBoxLayout()

        # Model path
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("Model Path:"))
        self.model_path_input = QLineEdit()
        path_layout.addWidget(self.model_path_input)

        browse_model_btn = QPushButton("Browse")
        browse_model_btn.clicked.connect(self.browse_model)
        path_layout.addWidget(browse_model_btn)

        load_model_btn = QPushButton("Load Model")
        load_model_btn.clicked.connect(self.load_model)
        path_layout.addWidget(load_model_btn)

        model_layout.addLayout(path_layout)

        # Model parameters
        params_layout = QHBoxLayout()

        params_layout.addWidget(QLabel("Threads:"))
        self.n_threads = QSpinBox()
        self.n_threads.setMinimum(1)
        self.n_threads.setMaximum(32)
        self.n_threads.setValue(4)
        params_layout.addWidget(self.n_threads)

        params_layout.addWidget(QLabel("Context:"))
        self.n_ctx = QSpinBox()
        self.n_ctx.setMinimum(128)
        self.n_ctx.setMaximum(32768)
        self.n_ctx.setValue(2048)
        params_layout.addWidget(self.n_ctx)

        params_layout.addWidget(QLabel("Temperature:"))
        self.temperature = QDoubleSpinBox()
        self.temperature.setMinimum(0.0)
        self.temperature.setMaximum(2.0)
        self.temperature.setSingleStep(0.1)
        self.temperature.setValue(0.7)
        params_layout.addWidget(self.temperature)

        params_layout.addWidget(QLabel("Max Tokens:"))
        self.max_tokens = QSpinBox()
        self.max_tokens.setMinimum(1)
        self.max_tokens.setMaximum(8192)
        self.max_tokens.setValue(512)
        params_layout.addWidget(self.max_tokens)

        model_layout.addLayout(params_layout)

        # Unload button
        unload_btn = QPushButton("Unload Model")
        unload_btn.clicked.connect(self.unload_model)
        model_layout.addWidget(unload_btn)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        layout.addStretch()

        return widget

    # Event Handlers
    def discover_models(self):
        """Discover models matching the task description"""
        task_desc = self.task_input.toPlainText().strip()
        category = self.task_category.currentText()

        if not task_desc:
            QMessageBox.warning(self, "Warning", "Please enter a task description")
            return

        matched = self.model_manager.find_matching_models(task_desc, category)

        if matched:
            self.matched_list.clear()
            for model in matched:
                self.matched_list.append(f"✓ {model['name']} - {model['type']} - Tags: {', '.join(model['tags'])}")
            self.statusBar().showMessage(f"Found {len(matched)} matching model(s)")
        else:
            self.matched_list.setText("No matching models found. Try adjusting your task description or add models to the registry.")
            self.statusBar().showMessage("No matches found")

    def load_registry(self):
        """Load and display model registry"""
        registry = self.model_manager.get_registry()
        self.registry_table.setRowCount(len(registry))

        for i, model in enumerate(registry):
            self.registry_table.setItem(i, 0, QTableWidgetItem(model["name"]))
            self.registry_table.setItem(i, 1, QTableWidgetItem(model["type"]))
            self.registry_table.setItem(i, 2, QTableWidgetItem(", ".join(model["tags"])))
            self.registry_table.setItem(i, 3, QTableWidgetItem(model["path"]))

    def add_model_to_registry(self):
        """Add a new model to the registry"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select GGUF Model", "", "GGUF Files (*.gguf);;All Files (*)"
        )

        if file_path:
            # Simple dialog to get model info (in production, use custom dialog)
            name = Path(file_path).stem
            self.model_manager.add_to_registry(name, file_path, "general", ["general"])
            self.load_registry()
            QMessageBox.information(self, "Success", f"Added {name} to registry")

    def browse_model(self):
        """Browse for a model file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select GGUF Model", "", "GGUF Files (*.gguf);;All Files (*)"
        )
        if file_path:
            self.model_path_input.setText(file_path)

    def load_model(self):
        """Load the selected model"""
        model_path = self.model_path_input.text().strip()
        if not model_path:
            QMessageBox.warning(self, "Warning", "Please select a model file")
            return

        try:
            self.statusBar().showMessage("Loading model...")
            self.model_manager.load_model(
                model_path,
                n_threads=self.n_threads.value(),
                n_ctx=self.n_ctx.value(),
                temperature=self.temperature.value(),
                max_tokens=self.max_tokens.value()
            )
            self.statusBar().showMessage("Model loaded successfully")
            QMessageBox.information(self, "Success", "Model loaded successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
            self.statusBar().showMessage("Error loading model")

    def unload_model(self):
        """Unload the current model"""
        self.model_manager.unload_model()
        self.statusBar().showMessage("Model unloaded")
        QMessageBox.information(self, "Info", "Model unloaded")

    def run_single_model(self):
        """Run prompt on a single model"""
        system = self.system_prompt.toPlainText().strip()
        user = self.user_prompt.toPlainText().strip()

        if not user:
            QMessageBox.warning(self, "Warning", "Please enter a user prompt")
            return

        if not self.model_manager.current_model:
            QMessageBox.warning(self, "Warning", "Please load a model first")
            return

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        self.run_single_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.statusBar().showMessage("Generating response...")

        # Run in worker thread
        self.worker = InferenceWorker(self.model_manager, messages, "current")
        self.worker.finished.connect(self.on_inference_finished)
        self.worker.error.connect(self.on_inference_error)
        self.worker.start()

    def on_inference_finished(self, result):
        """Handle inference completion"""
        self.response_viewer.setText(result["text"])
        self.stats_label.setText(
            f"Stats: {result['tokens']} tokens | {result['latency']:.2f}s | "
            f"{result['tokens']/result['latency']:.1f} tok/s"
        )
        self.run_single_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("Response generated")

    def on_inference_error(self, error_msg):
        """Handle inference error"""
        QMessageBox.critical(self, "Error", f"Inference failed: {error_msg}")
        self.run_single_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("Error")

    def run_all_models(self):
        """Run prompt on all matched models"""
        system = self.system_prompt.toPlainText().strip()
        user = self.user_prompt.toPlainText().strip()

        if not user:
            QMessageBox.warning(self, "Warning", "Please enter a user prompt")
            return

        task_desc = self.task_input.toPlainText().strip()
        if not task_desc:
            QMessageBox.warning(self, "Warning", "Please define a task first")
            return

        matched = self.model_manager.find_matching_models(task_desc, self.task_category.currentText())

        if not matched:
            QMessageBox.warning(self, "Warning", "No matching models found")
            return

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        self.statusBar().showMessage("Running on all models...")
        results = self.evaluator.run_on_multiple_models(messages, matched)

        self.display_comparison_results(results)
        self.statusBar().showMessage(f"Completed evaluation on {len(results)} model(s)")

    def display_comparison_results(self, results):
        """Display multi-model comparison results"""
        self.comparison_table.setRowCount(len(results))

        for i, result in enumerate(results):
            self.comparison_table.setItem(i, 0, QTableWidgetItem(result["model_name"]))

            snippet = result["text"][:100] + "..." if len(result["text"]) > 100 else result["text"]
            self.comparison_table.setItem(i, 1, QTableWidgetItem(snippet))

            self.comparison_table.setItem(i, 2, QTableWidgetItem(f"{result['latency']:.2f}"))
            self.comparison_table.setItem(i, 3, QTableWidgetItem(str(result["tokens"])))
            self.comparison_table.setItem(i, 4, QTableWidgetItem(result.get("pass_fail", "N/A")))
            self.comparison_table.setItem(i, 5, QTableWidgetItem(str(result.get("rating", "-"))))

    def browse_eval_file(self):
        """Browse for evaluation file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Eval File", "evals", "YAML Files (*.yaml *.yml);;All Files (*)"
        )
        if file_path:
            self.eval_file_input.setText(file_path)

    def run_evaluation(self):
        """Run evaluation on selected eval set"""
        eval_file = self.eval_file_input.text().strip()
        if not eval_file:
            QMessageBox.warning(self, "Warning", "Please select an eval file")
            return

        task_desc = self.task_input.toPlainText().strip()
        matched = self.model_manager.find_matching_models(task_desc, self.task_category.currentText())

        if not matched:
            QMessageBox.warning(self, "Warning", "No matching models found")
            return

        try:
            self.statusBar().showMessage("Running evaluation...")
            results = self.evaluator.run_eval_set(eval_file, matched)
            self.display_comparison_results(results)
            self.statusBar().showMessage(f"Evaluation complete: {len(results)} results")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Evaluation failed: {str(e)}")

    def export_results(self):
        """Export comparison results to CSV"""
        if self.comparison_table.rowCount() == 0:
            QMessageBox.warning(self, "Warning", "No results to export")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "", "CSV Files (*.csv);;All Files (*)"
        )

        if file_path:
            try:
                import csv
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)

                    # Headers
                    headers = [self.comparison_table.horizontalHeaderItem(i).text() 
                              for i in range(self.comparison_table.columnCount() - 1)]  # Exclude Actions
                    writer.writerow(headers)

                    # Data
                    for row in range(self.comparison_table.rowCount()):
                        row_data = [
                            self.comparison_table.item(row, col).text() if self.comparison_table.item(row, col) else ""
                            for col in range(self.comparison_table.columnCount() - 1)
                        ]
                        writer.writerow(row_data)

                QMessageBox.information(self, "Success", f"Results exported to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")

    def save_prompt(self):
        """Save current prompt to file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Prompt", "", "Markdown Files (*.md);;All Files (*)"
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"# System Prompt\n\n{self.system_prompt.toPlainText()}\n\n")
                    f.write(f"# User Prompt\n\n{self.user_prompt.toPlainText()}\n")
                QMessageBox.information(self, "Success", "Prompt saved")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Save failed: {str(e)}")

    def load_prompt(self):
        """Load prompt from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Prompt", "", "Markdown Files (*.md);;All Files (*)"
        )

        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Simple parsing
                    parts = content.split("# User Prompt")
                    if len(parts) == 2:
                        system_part = parts[0].replace("# System Prompt", "").strip()
                        user_part = parts[1].strip()
                        self.system_prompt.setText(system_part)
                        self.user_prompt.setText(user_part)
                    else:
                        self.user_prompt.setText(content)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Load failed: {str(e)}")

    def load_saved_state(self):
        """Load saved application state"""
        state = self.config_manager.load_state()
        if state:
            self.model_path_input.setText(state.get("last_model_path", ""))
            self.n_threads.setValue(state.get("n_threads", 4))
            self.n_ctx.setValue(state.get("n_ctx", 2048))
            self.temperature.setValue(state.get("temperature", 0.7))
            self.max_tokens.setValue(state.get("max_tokens", 512))

    def closeEvent(self, event):
        """Save state on close"""
        state = {
            "last_model_path": self.model_path_input.text(),
            "n_threads": self.n_threads.value(),
            "n_ctx": self.n_ctx.value(),
            "temperature": self.temperature.value(),
            "max_tokens": self.max_tokens.value()
        }
        self.config_manager.save_state(state)
        event.accept()
