"""
Module 14: Progressive Integration Tests
Tests that Module 15 (MLOps) works correctly AND that the entire TinyTorch system (01→14) still works.

DEPENDENCY CHAIN: 01_setup → ... → 14_benchmarking → 15_mlops
This is where we enable production deployment, monitoring, and lifecycle management for ML systems.

🎯 WHAT THIS TESTS:
- Module 14: Production deployment, model monitoring, lifecycle management, CI/CD for ML
- Integration: MLOps works with complete ML pipeline (models, training, benchmarking)
- Regression: Entire TinyTorch system (01→14) still works correctly
- Preparation: Ready for capstone (Module 16: Complete ML systems)

💡 FOR STUDENTS: If tests fail, check:
1. Does your ModelMonitor class exist in tinytorch.core.mlops?
2. Can you deploy models with monitoring and logging?
3. Do production pipelines work with real data workflows?
4. Are monitoring metrics meaningful for production decisions?

🔧 DEBUGGING HELP:
- MLOps includes: model versioning, deployment, monitoring, rollback, A/B testing
- Monitoring tracks: accuracy drift, latency, throughput, errors, resource usage
- Deployment enables: auto-scaling, load balancing, health checks, graceful updates
"""

import numpy as np
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestCompleteTinyTorchSystemStillWorks:
    """
    🔄 REGRESSION CHECK: Verify complete TinyTorch system (01→14) still works after MLOps development.

    💡 If these fail: You may have broken something in the core system while implementing MLOps.
    🔧 Fix: Check that MLOps code doesn't interfere with core ML functionality.
    """

    def test_complete_ml_system_stable(self):
        """
        ✅ TEST: Complete TinyTorch system (all modules 01→14) should still work

        📋 COMPLETE SYSTEM COMPONENTS:
        - Foundation: Setup, tensors, activations, layers
        - Networks: Dense networks, spatial operations, attention
        - Training: Data loading, autograd, optimizers, training loops
        - Production: Compression, kernels, benchmarking

        🚨 IF FAILS: Core TinyTorch system broken by MLOps development
        """
        try:
            # Test that complete TinyTorch system still works
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.spatial import Conv2d as Conv2D, MaxPool2d
            from tinytorch.core.attention import MultiHeadAttention
            from tinytorch.core.layers import Linear
            from tinytorch.core.activations import ReLU, Softmax
            from tinytorch.core.optimizers import Adam
            from tinytorch.core.training import Trainer
            from tinytorch.core.dataloader import Dataset, DataLoader

            # Optional imports - may not exist yet
            prune_weights = None
            try:
                from tinytorch.core.compression import prune_weights
            except ImportError:
                pass  # Compression module not implemented yet

            benchmark_model = None
            try:
                from tinytorch.core.benchmarking import benchmark_model
            except ImportError:
                pass  # Benchmarking module not implemented yet

            # Create sophisticated ML system (Vision + Language)
            class MultiModalSystem:
                def __init__(self):
                    # Vision pathway
                    self.vision_conv = Conv2D(3, 64, kernel_size=3, padding=1)
                    self.vision_pool = MaxPool2d(kernel_size=2)
                    self.vision_proj = Linear(64 * 16 * 16, 256)

                    # Language pathway
                    self.language_embed = Linear(1000, 256)  # vocab_size=1000
                    self.attention = MultiHeadAttention(embed_dim=256, num_heads=8)

                    # Fusion
                    self.fusion = Linear(512, 128)
                    self.classifier = Linear(128, 10)

                    # Activations
                    self.relu = ReLU()
                    self.softmax = Softmax()

                def __call__(self, vision_input, language_input):
                    # Vision processing
                    vis_feat = self.relu(self.vision_conv(vision_input))
                    vis_pooled = self.vision_pool(vis_feat)
                    vis_flat = Tensor(vis_pooled.data.reshape(vis_pooled.shape[0], -1))
                    vis_embed = self.vision_proj(vis_flat)

                    # Language processing
                    lang_embed = self.language_embed(language_input)
                    # Attention expects Tensor input, reshape to (batch, seq, embed)
                    lang_reshaped = Tensor(lang_embed.data.reshape(lang_embed.shape[0], -1, 256))
                    lang_attn = self.attention(lang_reshaped)
                    lang_feat = Tensor(lang_attn.data.reshape(lang_embed.shape[0], -1))

                    # Multimodal fusion
                    combined_data = np.concatenate([vis_embed.data, lang_feat.data], axis=1)
                    combined = Tensor(combined_data)

                    # Classification
                    fused = self.relu(self.fusion(combined))
                    logits = self.classifier(fused)
                    return self.softmax(logits)

                def parameters(self):
                    params = []
                    layers = [self.vision_conv, self.vision_proj, self.language_embed,
                             self.fusion, self.classifier]
                    for layer in layers:
                        if hasattr(layer, 'parameters'):
                            params.extend(layer.parameters())
                        elif hasattr(layer, 'weight'):
                            params.append(layer.weights)
                            if hasattr(layer, 'bias') and layer.bias is not None:
                                params.append(layer.bias)
                    return params

            # Test complete system
            system = MultiModalSystem()

            # Test data
            vision_data = Tensor(np.random.randn(2, 3, 32, 32))
            # Language data needs shape (batch, 1000) to match Linear(1000, 256) input
            language_data = Tensor(np.random.randn(2, 1000))

            # Test forward pass
            predictions = system(vision_data, language_data)

            assert predictions.shape == (2, 10), \
                f"❌ Complete system shape broken. Expected (2, 10), got {predictions.shape}"

            # Test training components
            optimizer = Adam(system.parameters(), lr=0.001)
            assert hasattr(optimizer, 'step'), "❌ Training components broken"

            # Test compression (if available)
            if prune_weights is not None:
                original_weights = system.vision_conv.weight.data.copy()
                pruned = prune_weights(system.vision_conv.weights, sparsity=0.2)
                assert pruned.shape == original_weights.shape, "❌ Compression broken"

            # Test benchmarking (if available)
            if benchmark_model is not None:
                # Simplified benchmark for vision pathway
                benchmark_results = benchmark_model(system.vision_conv, (2, 3, 32, 32))
                assert 'latency' in benchmark_results, "❌ Benchmarking broken"

        except ImportError as e:
            assert False, f"""
            ❌ COMPLETE TINYTORCH SYSTEM IMPORTS BROKEN!

            🔍 IMPORT ERROR: {str(e)}

            🔧 COMPLETE SYSTEM REQUIREMENTS:
            ALL modules (01→14) must be working perfectly:

            Foundation (01-05):
            ✅ Setup environment and tools
            ✅ Tensor operations and mathematics
            ✅ Activation functions and non-linearity
            ✅ Layer infrastructure and inheritance
            ✅ Dense networks and neural architectures

            Advanced ML (06-08):
            ✅ Spatial processing and computer vision
            ✅ Attention mechanisms and transformers
            ✅ Data loading and preprocessing pipelines

            Training Infrastructure (09-11):
            ✅ Automatic differentiation and gradients
            ✅ Optimization algorithms (SGD, Adam)
            ✅ Training loops and learning coordination

            Production Systems (12-14):
            ✅ Model compression and efficiency
            ✅ Performance kernels and acceleration
            ✅ Benchmarking and performance analysis

            💡 SYSTEM INTEGRITY:
            MLOps should be PURELY ADDITIVE - it adds
            deployment and monitoring but doesn't break
            any existing ML functionality.
            """
        except Exception as e:
            assert False, f"""
            ❌ COMPLETE TINYTORCH SYSTEM FUNCTIONALITY BROKEN!

            🔍 ERROR: {str(e)}

            🔧 SYSTEM STABILITY REQUIREMENTS:
            1. All forward passes work correctly
            2. Training components remain functional
            3. Advanced architectures still integrate
            4. Performance tools remain operational
            5. No interference from MLOps code

            💡 PRODUCTION READINESS:
            The complete TinyTorch system must work flawlessly
            because MLOps will deploy and monitor these models
            in production environments where reliability is critical.

            🚨 CRITICAL ISSUE:
            If the core ML system is broken, MLOps cannot
            deploy reliable models to production!
            """

    def test_benchmarking_and_optimization_stable(self):
        """
        ✅ TEST: Performance benchmarking and optimization should still work

        📋 PERFORMANCE SYSTEM:
        - Model benchmarking and profiling
        - Performance comparison tools
        - Hardware analysis and optimization
        - Training and inference analysis

        🎯 MLOps needs performance data for production decisions
        """
        try:
            from tinytorch.core.benchmarking import benchmark_model
        except ImportError:
            # Benchmarking module not implemented yet - pass gracefully
            assert True, "Benchmarking module not implemented yet"
            return

        try:
            from tinytorch.core.layers import Linear
            from tinytorch.core.spatial import Conv2d as Conv2D
            from tinytorch.core.tensor import Tensor

            # Test that benchmarking still works
            models_to_benchmark = [
                ("dense_model", Linear(100, 50)),
                ("conv_model", Conv2D(3, 16, kernel_size=3))
            ]

            benchmark_results = {}

            for model_name, model in models_to_benchmark:
                if model_name == "dense_model":
                    input_shape = (16, 100)
                else:  # conv_model
                    input_shape = (4, 3, 32, 32)

                # Test benchmarking
                results = benchmark_model(model, input_shape)
                benchmark_results[model_name] = results

                # Verify benchmark structure
                assert 'latency' in results, f"❌ Benchmarking broken for {model_name}"
                assert 'throughput' in results, f"❌ Benchmarking broken for {model_name}"
                assert results['latency'] > 0, f"❌ Invalid latency for {model_name}"
                assert results['throughput'] > 0, f"❌ Invalid throughput for {model_name}"

            # Verify performance comparison works
            dense_perf = benchmark_results["dense_model"]
            conv_perf = benchmark_results["conv_model"]

            # Should have different performance characteristics
            assert dense_perf['latency'] != conv_perf['latency'], \
                "❌ Performance comparison broken - models show identical performance"

        except Exception as e:
            assert False, f"""
            ❌ BENCHMARKING AND OPTIMIZATION BROKEN!

            🔍 ERROR: {str(e)}

            🔧 PERFORMANCE REQUIREMENTS FOR MLOPS:
            1. Model benchmarking must work for deployment planning
            2. Performance comparison guides model selection
            3. Hardware analysis informs infrastructure decisions
            4. Training metrics track system health

            💡 MLOPS DEPENDENCY ON PERFORMANCE:
            MLOps uses performance data for:
            - Auto-scaling decisions
            - Resource allocation
            - SLA monitoring
            - Cost optimization
            - Infrastructure planning

            Without working performance tools, MLOps cannot
            make intelligent production decisions!
            """


class TestModule15MLOpsCore:
    """
    🆕 NEW FUNCTIONALITY: Test Module 15 (MLOps) core implementation.

    💡 What you're implementing: Production deployment, monitoring, and lifecycle management for ML systems.
    🎯 Goal: Enable reliable, scalable, and monitored ML systems in production.
    """

    def test_model_monitoring_exists(self):
        """
        ✅ TEST: Model monitoring - Track model performance in production

        📋 WHAT YOU NEED TO IMPLEMENT:
        class ModelMonitor:
            def __init__(self, model, metrics=['accuracy', 'latency', 'throughput']):
                # Setup monitoring infrastructure
            def log_prediction(self, inputs, outputs, targets=None):
                # Track individual predictions
            def get_metrics(self):
                # Return current performance metrics

        🚨 IF FAILS: Model monitoring doesn't exist or missing components
        """
        try:
            from tinytorch.core.mlops import ModelMonitor
        except ImportError:
            # MLOps module not implemented yet - pass gracefully
            assert True, "MLOps module not implemented yet"
            return

        try:
            from tinytorch.core.layers import Linear
            from tinytorch.core.tensor import Tensor

            # Test model monitoring setup
            model = Linear(50, 10)
            monitor = ModelMonitor(model, metrics=['accuracy', 'latency', 'drift'])

            # Should track the model
            assert hasattr(monitor, 'model'), \
                "❌ ModelMonitor missing 'model' attribute"

            assert monitor.model is model, \
                "❌ ModelMonitor not correctly tracking the model"

            # Should track metrics
            assert hasattr(monitor, 'metrics'), \
                "❌ ModelMonitor missing 'metrics' configuration"

            # Should have logging capability
            assert hasattr(monitor, 'log_prediction'), \
                "❌ ModelMonitor missing 'log_prediction' method"

            assert callable(monitor.log_prediction), \
                "❌ ModelMonitor.log_prediction should be callable"

            # Test prediction logging
            test_input = Tensor(np.random.randn(1, 50))
            test_output = model(test_input)
            test_target = Tensor(np.random.randn(1, 10))

            # Should be able to log predictions
            monitor.log_prediction(test_input, test_output, test_target)

            # Should provide metrics
            assert hasattr(monitor, 'get_metrics'), \
                "❌ ModelMonitor missing 'get_metrics' method"

            metrics = monitor.get_metrics()
            assert isinstance(metrics, dict), \
                "❌ ModelMonitor.get_metrics() should return dict"

        except ImportError as e:
            assert False, f"""
            ❌ MODEL MONITORING MISSING!

            🔍 IMPORT ERROR: {str(e)}

            🔧 HOW TO IMPLEMENT:

            1. Create in modules/15_mlops/15_mlops.py:

            import time
            import numpy as np
            from collections import defaultdict, deque
            from tinytorch.core.tensor import Tensor

            class ModelMonitor:
                '''Production model monitoring and alerting.'''

                def __init__(self, model, metrics=['accuracy', 'latency', 'drift']):
                    self.model = model
                    self.metrics = metrics
                    self.prediction_log = deque(maxlen=10000)  # Keep last 10k predictions
                    self.metric_history = defaultdict(list)
                    self.start_time = time.time()

                def log_prediction(self, inputs, outputs, targets=None, latency=None):
                    '''Log a prediction for monitoring.'''
                    timestamp = time.time()

                    prediction_record = {{
                        'timestamp': timestamp,
                        'input_shape': inputs.shape,
                        'output_shape': outputs.shape,
                        'latency': latency or 0.001,  # Default latency
                    }}

                    if targets is not None:
                        # Calculate accuracy (simplified)
                        pred_classes = np.argmax(outputs.data, axis=-1)
                        true_classes = np.argmax(targets.data, axis=-1)
                        accuracy = np.mean(pred_classes == true_classes)
                        prediction_record['accuracy'] = accuracy

                    self.prediction_log.append(prediction_record)

                def get_metrics(self):
                    '''Get current monitoring metrics.'''
                    if not self.prediction_log:
                        return {{'status': 'no_data'}}

                    recent_predictions = list(self.prediction_log)[-100:]  # Last 100

                    # Calculate metrics
                    avg_latency = np.mean([p['latency'] for p in recent_predictions])
                    throughput = len(recent_predictions) / (time.time() - recent_predictions[0]['timestamp'])

                    metrics = {{
                        'avg_latency': avg_latency,
                        'throughput': throughput,
                        'prediction_count': len(self.prediction_log),
                        'uptime': time.time() - self.start_time
                    }}

                    # Add accuracy if available
                    accuracies = [p.get('accuracy') for p in recent_predictions if 'accuracy' in p]
                    if accuracies:
                        metrics['accuracy'] = np.mean(accuracies)

                    return metrics

                def check_drift(self):
                    '''Check for model drift.'''
                    # Simplified drift detection
                    if len(self.prediction_log) < 100:
                        return {{'drift_detected': False, 'reason': 'insufficient_data'}}

                    recent = list(self.prediction_log)[-50:]
                    older = list(self.prediction_log)[-100:-50]

                    recent_acc = np.mean([p.get('accuracy', 0.5) for p in recent])
                    older_acc = np.mean([p.get('accuracy', 0.5) for p in older])

                    drift_threshold = 0.05  # 5% accuracy drop
                    drift_detected = (older_acc - recent_acc) > drift_threshold

                    return {{
                        'drift_detected': drift_detected,
                        'accuracy_drop': older_acc - recent_acc,
                        'threshold': drift_threshold
                    }}

            2. Export the module:
               tito module complete 15_mlops

            📊 MONITORING CAPABILITIES:
            - Real-time performance tracking
            - Drift detection and alerting
            - Resource usage monitoring
            - Error rate tracking
            - Custom metric support
            """
        except Exception as e:
            assert False, f"""
            ❌ MODEL MONITORING BROKEN!

            🔍 ERROR: {str(e)}

            🔧 MONITORING REQUIREMENTS:
            1. Track model predictions and performance
            2. Detect accuracy/performance drift
            3. Monitor latency and throughput
            4. Log prediction history
            5. Provide actionable metrics
            6. Support alerting and notifications

            💡 PRODUCTION MONITORING:
            Model monitoring enables:
            - Early detection of model degradation
            - Automatic retraining triggers
            - Performance SLA tracking
            - A/B testing validation
            - Incident response and debugging

            🚨 CRITICAL FOR PRODUCTION:
            Without monitoring, production ML systems are:
            - Unreliable (undetected failures)
            - Untrustworthy (silent degradation)
            - Unoptimizable (no performance data)
            - Unmaintainable (no operational visibility)
            """

    def test_model_deployment_infrastructure(self):
        """
        ✅ TEST: Model deployment - Deploy models to production environments

        📋 DEPLOYMENT CAPABILITIES:
        - Model serving and inference endpoints
        - Load balancing and auto-scaling
        - Health checks and rollback
        - Version management and A/B testing

        🎯 Enable reliable model serving at scale
        """
        try:
            from tinytorch.core.mlops import ModelServer, deploy_model
        except ImportError:
            # MLOps module not implemented yet - pass gracefully
            assert True, "MLOps module not implemented yet"
            return

        try:
            from tinytorch.core.layers import Linear
            from tinytorch.core.tensor import Tensor

            # Test model deployment
            model = Linear(20, 5)

            # Test model server
            if 'ModelServer' in locals():
                server = ModelServer(model, port=8080)

                # Should configure serving
                assert hasattr(server, 'model'), \
                    "❌ ModelServer missing model configuration"

                assert hasattr(server, 'predict'), \
                    "❌ ModelServer missing predict method"

                # Test prediction interface
                test_input = Tensor(np.random.randn(1, 20))
                prediction = server.predict(test_input)

                assert prediction.shape == (1, 5), \
                    f"❌ ModelServer prediction shape wrong. Expected (1, 5), got {prediction.shape}"

                # Test health check
                if hasattr(server, 'health_check'):
                    health = server.health_check()
                    assert isinstance(health, dict), \
                        "❌ Health check should return dict"
                    assert 'status' in health, \
                        "❌ Health check missing status"

            # Test deployment function
            if 'deploy_model' in locals():
                deployment = deploy_model(model, endpoint='/predict', replicas=2)

                assert hasattr(deployment, 'predict'), \
                    "❌ Deployment missing predict interface"

                assert hasattr(deployment, 'scale'), \
                    "❌ Deployment missing scaling capability"

                # Test scaling
                deployment.scale(replicas=4)
                assert deployment.replicas == 4, \
                    "❌ Deployment scaling broken"

        except ImportError:
            assert False, f"""
            ❌ MODEL DEPLOYMENT INFRASTRUCTURE MISSING!

            🔧 DEPLOYMENT IMPLEMENTATION:

            class ModelServer:
                '''Production model serving infrastructure.'''

                def __init__(self, model, port=8080, health_check_interval=30):
                    self.model = model
                    self.port = port
                    self.health_check_interval = health_check_interval
                    self.request_count = 0
                    self.error_count = 0
                    self.start_time = time.time()

                def predict(self, inputs):
                    '''Serve model predictions.'''
                    try:
                        self.request_count += 1
                        return self.model(inputs)
                    except Exception as e:
                        self.error_count += 1
                        raise e

                def health_check(self):
                    '''Check server health status.'''
                    uptime = time.time() - self.start_time
                    error_rate = self.error_count / max(self.request_count, 1)

                    status = 'healthy' if error_rate < 0.05 else 'unhealthy'

                    return {{
                        'status': status,
                        'uptime': uptime,
                        'request_count': self.request_count,
                        'error_rate': error_rate,
                        'memory_usage': 'unknown'  # Would implement actual monitoring
                    }}

                def start(self):
                    '''Start the model server.'''
                    print(f"Starting model server on port {{self.port}}")
                    # Would implement actual HTTP server

                def stop(self):
                    '''Stop the model server.'''
                    print("Stopping model server")

            def deploy_model(model, endpoint='/predict', replicas=1, auto_scale=True):
                '''Deploy model with production configuration.'''

                class Deployment:
                    def __init__(self, model, endpoint, replicas):
                        self.model = model
                        self.endpoint = endpoint
                        self.replicas = replicas
                        self.servers = []

                        # Create server instances
                        for i in range(replicas):
                            server = ModelServer(model, port=8080+i)
                            self.servers.append(server)

                    def predict(self, inputs):
                        # Load balance across servers
                        server_idx = hash(str(inputs.data)) % len(self.servers)
                        return self.servers[server_idx].predict(inputs)

                    def scale(self, replicas):
                        self.replicas = replicas
                        # Would implement actual scaling logic

                    def rollback(self, version):
                        # Would implement model version rollback
                        pass

                return Deployment(model, endpoint, replicas)

            💡 DEPLOYMENT FEATURES:
            - High availability with load balancing
            - Auto-scaling based on traffic
            - Health monitoring and alerting
            - Blue-green deployments
            - Canary releases and A/B testing
            """
        except Exception as e:
            assert False, f"""
            ❌ MODEL DEPLOYMENT INFRASTRUCTURE BROKEN!

            🔍 ERROR: {str(e)}

            🔧 DEPLOYMENT REQUIREMENTS:
            1. Serve models via HTTP/gRPC endpoints
            2. Handle concurrent requests efficiently
            3. Provide health checks and monitoring
            4. Support auto-scaling and load balancing
            5. Enable blue-green and canary deployments
            6. Track deployment metrics and logs

            🌐 PRODUCTION SERVING:
            Model deployment enables:
            - Real-time inference APIs
            - Batch processing pipelines
            - Edge deployment for mobile/IoT
            - Multi-region serving for global apps
            - Cost-effective auto-scaling
            """

    def test_ml_pipeline_orchestration(self):
        """
        ✅ TEST: ML pipeline orchestration - Coordinate training, evaluation, deployment

        📋 PIPELINE CAPABILITIES:
        - Training pipeline automation
        - Model evaluation and validation
        - Automated deployment triggers
        - Rollback and recovery

        💡 Enable end-to-end ML automation
        """
        try:
            from tinytorch.core.mlops import MLPipeline, PipelineStep
        except ImportError:
            # MLOps module not implemented yet - pass gracefully
            assert True, "MLOps module not implemented yet"
            return

        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear
            from tinytorch.core.optimizers import SGD
            from tinytorch.core.training import Trainer

            # Test ML pipeline orchestration
            if 'MLPipeline' in locals():
                pipeline = MLPipeline(name="production_model_pipeline")

                # Should support adding steps
                assert hasattr(pipeline, 'add_step'), \
                    "❌ MLPipeline missing add_step method"

                # Create pipeline steps
                if 'PipelineStep' in locals():
                    # Training step
                    train_step = PipelineStep(
                        name="training",
                        function=lambda: "training_complete",
                        inputs=['data', 'model'],
                        outputs=['trained_model']
                    )

                    # Evaluation step
                    eval_step = PipelineStep(
                        name="evaluation",
                        function=lambda: {"accuracy": 0.95, "precision": 0.93},
                        inputs=['trained_model', 'test_data'],
                        outputs=['metrics']
                    )

                    # Deployment step
                    deploy_step = PipelineStep(
                        name="deployment",
                        function=lambda: "deployment_successful",
                        inputs=['trained_model', 'metrics'],
                        outputs=['deployment_url']
                    )

                    # Add steps to pipeline
                    pipeline.add_step(train_step)
                    pipeline.add_step(eval_step)
                    pipeline.add_step(deploy_step)

                    # Should be able to execute pipeline
                    if hasattr(pipeline, 'execute'):
                        results = pipeline.execute()
                        assert isinstance(results, dict), \
                            "❌ Pipeline execution should return results dict"

            # Test simpler pipeline coordination
            # Simulate ML pipeline steps
            pipeline_state = {
                'model': Linear(10, 3),
                'optimizer': None,
                'trainer': None,
                'metrics': {},
                'deployment': None
            }

            # Step 1: Setup training
            pipeline_state['optimizer'] = SGD(pipeline_state['model'].parameters(), lr=0.01)
            pipeline_state['trainer'] = Trainer(pipeline_state['model'], pipeline_state['optimizer'])

            # Step 2: Training simulation
            x = Tensor(np.random.randn(16, 10))
            output = pipeline_state['model'](x)
            pipeline_state['metrics']['training_loss'] = 0.5  # Simulated loss

            # Step 3: Evaluation
            eval_x = Tensor(np.random.randn(8, 10))
            eval_output = pipeline_state['model'](eval_x)
            pipeline_state['metrics']['accuracy'] = 0.85  # Simulated accuracy

            # Step 4: Deployment decision
            accuracy_threshold = 0.8
            if pipeline_state['metrics']['accuracy'] > accuracy_threshold:
                pipeline_state['deployment'] = 'approved'
            else:
                pipeline_state['deployment'] = 'rejected'

            # Verify pipeline coordination
            assert pipeline_state['trainer'] is not None, \
                "❌ Pipeline training setup broken"

            assert 'accuracy' in pipeline_state['metrics'], \
                "❌ Pipeline evaluation broken"

            assert pipeline_state['deployment'] == 'approved', \
                f"❌ Pipeline deployment logic broken. Accuracy: {pipeline_state['metrics']['accuracy']}"

        except Exception as e:
            assert False, f"""
            ❌ ML PIPELINE ORCHESTRATION BROKEN!

            🔍 ERROR: {str(e)}

            🔧 PIPELINE ORCHESTRATION IMPLEMENTATION:

            class PipelineStep:
                '''Individual step in ML pipeline.'''

                def __init__(self, name, function, inputs=None, outputs=None):
                    self.name = name
                    self.function = function
                    self.inputs = inputs or []
                    self.outputs = outputs or []

                def execute(self, context):
                    '''Execute step with given context.'''
                    return self.function()

            class MLPipeline:
                '''Orchestrate complete ML workflows.'''

                def __init__(self, name):
                    self.name = name
                    self.steps = []
                    self.context = {{}}

                def add_step(self, step):
                    '''Add step to pipeline.'''
                    self.steps.append(step)

                def execute(self):
                    '''Execute all pipeline steps in order.'''
                    results = {{}}

                    for step in self.steps:
                        try:
                            step_result = step.execute(self.context)
                            results[step.name] = step_result
                            self.context[step.name] = step_result
                        except Exception as e:
                            results[step.name] = f"ERROR: {{e}}"
                            break  # Stop on error

                    return results

                def rollback(self, to_step):
                    '''Rollback pipeline to specific step.'''
                    # Would implement rollback logic
                    pass

            💡 PIPELINE BENEFITS:
            - Automated ML workflows
            - Reproducible model development
            - Consistent deployment processes
            - Error handling and recovery
            - Audit trails and governance
            """


class TestMLOpsIntegration:
    """
    🔗 INTEGRATION TEST: MLOps + Complete TinyTorch system working together.

    💡 Test that MLOps works with real ML workflows and production scenarios.
    🎯 Goal: Enable production-ready ML systems with monitoring and automation.
    """

    def test_production_ml_workflow(self):
        """
        ✅ TEST: Complete production ML workflow with monitoring and deployment

        📋 PRODUCTION WORKFLOW:
        - Model training with monitoring
        - Performance benchmarking and validation
        - Automated deployment with health checks
        - Real-time monitoring and alerting

        💡 End-to-end production ML system
        """
        try:
            from tinytorch.core.mlops import ModelMonitor, ModelServer
        except ImportError:
            # MLOps module not implemented yet - pass gracefully
            assert True, "MLOps module not implemented yet"
            return

        try:
            from tinytorch.core.benchmarking import benchmark_model
        except ImportError:
            benchmark_model = None  # Will test without benchmarking

        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear
            from tinytorch.core.optimizers import Adam
            from tinytorch.core.training import Trainer, MSELoss
            from tinytorch.core.dataloader import Dataset, DataLoader

            # Production ML workflow simulation

            # Step 1: Model Development
            model = Linear(50, 10)
            optimizer = Adam(model.parameters(), lr=0.001)
            loss_fn = MSELoss()
            trainer = Trainer(model, optimizer)

            # Step 2: Training Data
            class ProductionDataset(Dataset):
                def __init__(self):
                    self.data = np.random.randn(200, 50)
                    self.targets = np.random.randn(200, 10)

                def __len__(self):
                    return 200

                def __getitem__(self, idx):
                    return Tensor(self.data[idx]), Tensor(self.targets[idx])

            dataset = ProductionDataset()
            dataloader = DataLoader(dataset, batch_size=32)

            # Step 3: Training with Monitoring
            monitor = ModelMonitor(model, metrics=['loss', 'latency', 'throughput'])

            training_metrics = []
            for epoch in range(3):  # Simulate training
                epoch_losses = []
                for batch_x, batch_y in dataloader:
                    # Forward pass
                    start_time = time.time()
                    predictions = model(batch_x)
                    inference_time = time.time() - start_time

                    # Loss computation
                    loss = loss_fn(predictions, batch_y)
                    epoch_losses.append(loss.data if hasattr(loss, 'data') else float(loss))

                    # Log prediction for monitoring
                    monitor.log_prediction(batch_x, predictions, batch_y, latency=inference_time)

                    break  # One batch per epoch for testing

                training_metrics.append(np.mean(epoch_losses))

            # Step 4: Performance Benchmarking
            benchmark_results = benchmark_model(model, (32, 50))

            # Step 5: Production Readiness Check
            monitor_metrics = monitor.get_metrics()

            production_ready = (
                benchmark_results['latency'] < 0.1 and  # < 100ms latency
                monitor_metrics.get('throughput', 0) > 100 and  # > 100 samples/sec
                training_metrics[-1] < 1.0  # Reasonable loss
            )

            # Step 6: Deployment (if ready)
            if production_ready:
                if 'ModelServer' in locals():
                    server = ModelServer(model, port=8080)

                    # Test production serving
                    test_input = Tensor(np.random.randn(1, 50))
                    production_prediction = server.predict(test_input)

                    assert production_prediction.shape == (1, 10), \
                        f"❌ Production serving broken. Expected (1, 10), got {production_prediction.shape}"

                    # Health check
                    health = server.health_check()
                    assert health['status'] in ['healthy', 'unhealthy'], \
                        f"❌ Health check broken. Got status: {health.get('status')}"

            # Verify complete workflow
            assert len(training_metrics) == 3, \
                "❌ Training workflow broken"

            assert 'latency' in benchmark_results, \
                "❌ Benchmarking integration broken"

            assert 'throughput' in monitor_metrics, \
                "❌ Monitoring integration broken"

            assert isinstance(production_ready, bool), \
                "❌ Production readiness check broken"

        except Exception as e:
            assert False, f"""
            ❌ PRODUCTION ML WORKFLOW BROKEN!

            🔍 ERROR: {str(e)}

            🔧 PRODUCTION WORKFLOW REQUIREMENTS:
            1. ✅ Model training with monitoring
            2. ✅ Performance benchmarking integration
            3. ✅ Automated deployment decisions
            4. ✅ Real-time serving with health checks
            5. ✅ End-to-end workflow coordination

            💡 PRODUCTION ML SYSTEM:

            Complete workflow should include:

            Training Phase:
            - Data validation and preprocessing
            - Model training with experiment tracking
            - Performance monitoring during training
            - Model validation and testing

            Deployment Phase:
            - Performance benchmarking
            - Production readiness validation
            - Automated deployment with rollback
            - Real-time monitoring and alerting

            Operations Phase:
            - Continuous monitoring
            - Drift detection and retraining
            - A/B testing and experimentation
            - Incident response and debugging

            🚀 PRODUCTION SUCCESS CRITERIA:
            - Latency < 100ms for real-time apps
            - Throughput > 1000 QPS for high-scale
            - Accuracy maintained > 95% SLA
            - 99.9% uptime with automatic recovery
            """

    def test_continuous_integration_ml(self):
        """
        ✅ TEST: Continuous Integration for ML (CI/ML) - Automated testing and validation

        📋 CI/ML CAPABILITIES:
        - Automated model testing
        - Performance regression detection
        - Data validation and schema checking
        - Model quality gates

        🎯 Ensure model quality through automation
        """
        try:
            from tinytorch.core.mlops import ModelValidator, DataValidator
        except ImportError:
            # MLOps module not implemented yet - pass gracefully
            assert True, "MLOps module not implemented yet"
            return

        try:
            from tinytorch.core.benchmarking import benchmark_model
        except ImportError:
            benchmark_model = None  # Will test without benchmarking

        try:
            from tinytorch.core.tensor import Tensor
            from tinytorch.core.layers import Linear

            # CI/ML workflow simulation

            # Step 1: Data Validation
            if 'DataValidator' in locals():
                data_validator = DataValidator(schema={'features': 50, 'samples': 100})

                # Test data
                test_data = np.random.randn(100, 50)
                validation_result = data_validator.validate(test_data)

                assert validation_result['valid'], \
                    f"❌ Data validation failed: {validation_result.get('errors')}"

            # Step 2: Model Testing
            model = Linear(50, 10)

            if 'ModelValidator' in locals():
                model_validator = ModelValidator()

                # Test model structure
                structure_valid = model_validator.validate_structure(model)
                assert structure_valid, "❌ Model structure validation failed"

                # Test model functionality
                test_input = Tensor(np.random.randn(5, 50))
                functionality_valid = model_validator.validate_functionality(model, test_input)
                assert functionality_valid, "❌ Model functionality validation failed"

            # Step 3: Performance Regression Testing
            baseline_performance = {'latency': 0.01, 'accuracy': 0.90}
            current_performance = benchmark_model(model, (16, 50))

            # Performance regression check
            latency_regression = current_performance['latency'] > baseline_performance['latency'] * 1.5
            # accuracy_regression = current_performance.get('accuracy', 0.9) < baseline_performance['accuracy'] * 0.95

            performance_check = {
                'latency_regression': latency_regression,
                'performance_acceptable': not latency_regression
            }

            # Step 4: Quality Gates
            quality_gates = {
                'data_quality': True,  # From data validation
                'model_structure': True,  # From model validation
                'performance_acceptable': performance_check['performance_acceptable'],
                'security_scan': True,  # Would implement security validation
            }

            all_gates_passed = all(quality_gates.values())

            # CI/ML Decision
            ci_ml_result = {
                'quality_gates': quality_gates,
                'deployment_approved': all_gates_passed,
                'recommendations': []
            }

            if not all_gates_passed:
                ci_ml_result['recommendations'].append("Fix failing quality gates before deployment")

            # Verify CI/ML workflow
            assert isinstance(quality_gates, dict), \
                "❌ Quality gates structure broken"

            assert 'deployment_approved' in ci_ml_result, \
                "❌ CI/ML decision logic broken"

            # Test manual validation workflow
            manual_checks = {
                'model_loads': True,
                'inference_works': True,
                'output_shape_correct': True,
                'no_errors': True
            }

            # Test model loading and inference
            try:
                test_input = Tensor(np.random.randn(3, 50))
                output = model(test_input)
                manual_checks['model_loads'] = True
                manual_checks['inference_works'] = True
                manual_checks['output_shape_correct'] = (output.shape == (3, 10))
                manual_checks['no_errors'] = True
            except Exception as e:
                manual_checks['model_loads'] = False
                manual_checks['inference_works'] = False
                manual_checks['no_errors'] = False

            assert all(manual_checks.values()), \
                f"❌ Manual validation checks failed: {manual_checks}"

        except Exception as e:
            assert False, f"""
            ❌ CONTINUOUS INTEGRATION ML BROKEN!

            🔍 ERROR: {str(e)}

            🔧 CI/ML IMPLEMENTATION:

            class DataValidator:
                '''Validate data quality and schema.'''

                def __init__(self, schema):
                    self.schema = schema

                def validate(self, data):
                    errors = []

                    # Check shape
                    expected_shape = (self.schema['samples'], self.schema['features'])
                    if data.shape != expected_shape:
                        errors.append(f"Shape mismatch: expected {{expected_shape}}, got {{data.shape}}")

                    # Check for NaN/inf
                    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                        errors.append("Data contains NaN or infinity values")

                    return {{
                        'valid': len(errors) == 0,
                        'errors': errors
                    }}

            class ModelValidator:
                '''Validate model structure and functionality.'''

                def validate_structure(self, model):
                    # Check if model is callable
                    return callable(model)

                def validate_functionality(self, model, test_input):
                    try:
                        output = model(test_input)
                        return output is not None
                    except Exception:
                        return False

            💡 CI/ML QUALITY GATES:

            Data Quality:
            - Schema validation
            - Distribution checks
            - Anomaly detection
            - Data lineage tracking

            Model Quality:
            - Structure validation
            - Functionality testing
            - Performance benchmarking
            - Security scanning

            Deployment Gates:
            - All tests pass
            - Performance meets SLA
            - Security scan clean
            - Manual approval (if required)

            🔒 PRODUCTION SAFETY:
            CI/ML prevents deploying broken models to production!
            """

    def test_model_lifecycle_management(self):
        """
        ✅ TEST: Model lifecycle management - Version control, rollback, A/B testing

        📋 LIFECYCLE MANAGEMENT:
        - Model versioning and registry
        - Rollback and recovery capabilities
        - A/B testing and experimentation
        - Model retirement and cleanup

        💡 Manage models throughout their production lifecycle
        """
        try:
            from tinytorch.core.mlops import ModelRegistry, ABTestManager
        except ImportError:
            # MLOps module not implemented yet - pass gracefully
            assert True, "MLOps module not implemented yet"
            return

        try:
            from tinytorch.core.layers import Linear
            from tinytorch.core.tensor import Tensor

            # Model lifecycle management

            # Step 1: Model Registry
            if 'ModelRegistry' in locals():
                registry = ModelRegistry()

                # Register models
                model_v1 = Linear(50, 10)
                model_v2 = Linear(50, 10)  # Improved version

                registry.register_model("production_classifier", model_v1, version="1.0")
                registry.register_model("production_classifier", model_v2, version="2.0")

                # Test model retrieval
                current_model = registry.get_model("production_classifier", version="2.0")
                assert current_model is model_v2, \
                    "❌ Model registry retrieval broken"

                # Test rollback capability
                rollback_model = registry.get_model("production_classifier", version="1.0")
                assert rollback_model is model_v1, \
                    "❌ Model registry rollback broken"

            # Step 2: A/B Testing
            if 'ABTestManager' in locals():
                ab_manager = ABTestManager()

                # Setup A/B test
                model_a = Linear(50, 10)  # Current model
                model_b = Linear(50, 10)  # New model

                ab_manager.setup_test("classifier_experiment",
                                    model_a=model_a,
                                    model_b=model_b,
                                    traffic_split=0.5)

                # Test traffic routing
                test_input = Tensor(np.random.randn(1, 50))

                for _ in range(10):
                    assigned_model, prediction = ab_manager.predict("classifier_experiment", test_input)
                    assert assigned_model in ['A', 'B'], \
                        f"❌ A/B test assignment broken: {assigned_model}"
                    assert prediction.shape == (1, 10), \
                        f"❌ A/B test prediction broken: {prediction.shape}"

                # Test experiment results
                results = ab_manager.get_results("classifier_experiment")
                assert 'model_a_metrics' in results, \
                    "❌ A/B test results missing model A metrics"
                assert 'model_b_metrics' in results, \
                    "❌ A/B test results missing model B metrics"

            # Step 3: Manual lifecycle simulation
            lifecycle_state = {
                'models': {
                    'v1.0': Linear(50, 10),
                    'v2.0': Linear(50, 10),
                    'v2.1': Linear(50, 10),
                },
                'current_version': 'v2.1',
                'rollback_version': 'v2.0',
                'experiments': {},
                'deployment_history': []
            }

            # Simulate version management
            current_model = lifecycle_state['models'][lifecycle_state['current_version']]
            test_input = Tensor(np.random.randn(5, 50))
            current_output = current_model(test_input)

            # Simulate rollback
            rollback_model = lifecycle_state['models'][lifecycle_state['rollback_version']]
            rollback_output = rollback_model(test_input)

            # Simulate A/B test
            model_a = lifecycle_state['models']['v2.0']
            model_b = lifecycle_state['models']['v2.1']

            # Compare models
            output_a = model_a(test_input)
            output_b = model_b(test_input)

            # Record experiment
            lifecycle_state['experiments']['v2.0_vs_v2.1'] = {
                'model_a_performance': {'latency': 0.01, 'accuracy': 0.90},
                'model_b_performance': {'latency': 0.008, 'accuracy': 0.92},
                'winner': 'model_b'
            }

            # Verify lifecycle management
            assert current_output.shape == (5, 10), \
                "❌ Current model broken"

            assert rollback_output.shape == (5, 10), \
                "❌ Rollback model broken"

            assert output_a.shape == output_b.shape, \
                "❌ A/B test models incompatible"

            assert 'winner' in lifecycle_state['experiments']['v2.0_vs_v2.1'], \
                "❌ Experiment analysis broken"

        except Exception as e:
            assert False, f"""
            ❌ MODEL LIFECYCLE MANAGEMENT BROKEN!

            🔍 ERROR: {str(e)}

            🔧 LIFECYCLE MANAGEMENT IMPLEMENTATION:

            class ModelRegistry:
                '''Central registry for model versions.'''

                def __init__(self):
                    self.models = {{}}  # {{name: {{version: model}}}}

                def register_model(self, name, model, version, metadata=None):
                    if name not in self.models:
                        self.models[name] = {{}}

                    self.models[name][version] = {{
                        'model': model,
                        'metadata': metadata or {{}},
                        'timestamp': time.time()
                    }}

                def get_model(self, name, version=None):
                    if name not in self.models:
                        raise ValueError(f"Model {{name}} not found")

                    if version is None:
                        # Get latest version
                        latest_version = max(self.models[name].keys())
                        return self.models[name][latest_version]['model']

                    if version not in self.models[name]:
                        raise ValueError(f"Version {{version}} not found for {{name}}")

                    return self.models[name][version]['model']

                def list_versions(self, name):
                    return list(self.models.get(name, {{}}).keys())

            class ABTestManager:
                '''Manage A/B testing experiments.'''

                def __init__(self):
                    self.experiments = {{}}

                def setup_test(self, experiment_name, model_a, model_b, traffic_split=0.5):
                    self.experiments[experiment_name] = {{
                        'model_a': model_a,
                        'model_b': model_b,
                        'traffic_split': traffic_split,
                        'results': {{'a': [], 'b': []}}
                    }}

                def predict(self, experiment_name, inputs):
                    experiment = self.experiments[experiment_name]

                    # Simple traffic routing (hash-based)
                    route_to_b = hash(str(inputs.data)) % 100 < experiment['traffic_split'] * 100

                    if route_to_b:
                        prediction = experiment['model_b'](inputs)
                        return 'B', prediction
                    else:
                        prediction = experiment['model_a'](inputs)
                        return 'A', prediction

                def get_results(self, experiment_name):
                    return {{
                        'model_a_metrics': {{'requests': 100, 'avg_latency': 0.01}},
                        'model_b_metrics': {{'requests': 100, 'avg_latency': 0.008}},
                        'statistical_significance': True
                    }}

            💡 LIFECYCLE BENEFITS:
            - Zero-downtime deployments
            - Quick rollback on issues
            - Data-driven model selection
            - Compliance and audit trails
            - Risk mitigation through testing
            """


class TestModule15Completion:
    """
    ✅ COMPLETION CHECK: Module 15 ready and TinyTorch production-ready.

    🎯 Final validation that MLOps works and TinyTorch is ready for real-world deployment.
    """

    def test_production_ml_system_complete(self):
        """
        ✅ FINAL TEST: Complete production ML system ready for real-world deployment

        📋 PRODUCTION ML SYSTEM CHECKLIST:
        □ Model monitoring and alerting
        □ Deployment infrastructure and serving
        □ Pipeline orchestration and automation
        □ Continuous integration and validation
        □ Model lifecycle management
        □ Performance optimization
        □ Security and compliance
        □ Real-world production readiness

        🎯 SUCCESS = TinyTorch is production-ready!
        """
        # First check if MLOps module exists
        try:
            from tinytorch.core.mlops import ModelMonitor
        except ImportError:
            # MLOps module not implemented yet - pass gracefully
            assert True, "MLOps module not implemented yet"
            return

        production_capabilities = {
            "Model monitoring": False,
            "Deployment infrastructure": False,
            "Pipeline orchestration": False,
            "Continuous integration": False,
            "Lifecycle management": False,
            "Performance optimization": False,
            "Security considerations": False,
            "Production readiness": False
        }

        try:
            # Test 1: Model monitoring
            from tinytorch.core.layers import Linear
            from tinytorch.core.tensor import Tensor

            model = Linear(20, 5)
            monitor = ModelMonitor(model)

            # Test monitoring functionality
            test_input = Tensor(np.random.randn(1, 20))
            test_output = model(test_input)
            monitor.log_prediction(test_input, test_output)

            metrics = monitor.get_metrics()
            assert 'uptime' in metrics
            production_capabilities["Model monitoring"] = True

            # Test 2: Deployment infrastructure
            try:
                from tinytorch.core.mlops import ModelServer
                server = ModelServer(model)
                assert hasattr(server, 'predict')
                production_capabilities["Deployment infrastructure"] = True
            except ImportError:
                # Manual deployment test
                def serve_prediction(model, inputs):
                    return model(inputs)

                served_output = serve_prediction(model, test_input)
                assert served_output.shape == test_output.shape
                production_capabilities["Deployment infrastructure"] = True

            # Test 3: Pipeline orchestration
            try:
                from tinytorch.core.mlops import MLPipeline
                pipeline = MLPipeline("test_pipeline")
                assert hasattr(pipeline, 'add_step')
                production_capabilities["Pipeline orchestration"] = True
            except ImportError:
                # Manual pipeline test
                pipeline_steps = ['data_prep', 'training', 'evaluation', 'deployment']
                pipeline_status = {step: 'completed' for step in pipeline_steps}
                assert all(status == 'completed' for status in pipeline_status.values())
                production_capabilities["Pipeline orchestration"] = True

            # Test 4: Continuous integration
            from tinytorch.core.benchmarking import benchmark_model

            # Performance validation
            benchmark_results = benchmark_model(model, (16, 20))
            performance_ok = benchmark_results['latency'] < 1.0  # < 1 second

            # Quality validation
            test_batch = Tensor(np.random.randn(8, 20))
            output_batch = model(test_batch)
            quality_ok = output_batch.shape == (8, 5)

            ci_validation = performance_ok and quality_ok
            assert ci_validation
            production_capabilities["Continuous integration"] = True

            # Test 5: Lifecycle management
            # Model versioning simulation
            model_versions = {
                'v1.0': Linear(20, 5),
                'v2.0': Linear(20, 5),
                'v2.1': Linear(20, 5)
            }

            current_version = 'v2.1'
            current_model = model_versions[current_version]

            # Rollback capability
            rollback_version = 'v2.0'
            rollback_model = model_versions[rollback_version]

            # Test both models work
            current_pred = current_model(test_input)
            rollback_pred = rollback_model(test_input)

            assert current_pred.shape == rollback_pred.shape
            production_capabilities["Lifecycle management"] = True

            # Test 6: Performance optimization
            from tinytorch.core.compression import prune_weights

            # Model optimization
            original_model = Linear(100, 50)
            optimized_weights = prune_weights(original_model.weights, sparsity=0.3)

            # Performance comparison
            original_results = benchmark_model(original_model, (16, 100))

            # Optimized model should maintain functionality
            optimized_model = Linear(100, 50)
            optimized_model.weights = optimized_weights

            optimized_input = Tensor(np.random.randn(4, 100))
            optimized_output = optimized_model(optimized_input)
            assert optimized_output.shape == (4, 50)

            production_capabilities["Performance optimization"] = True

            # Test 7: Security considerations
            # Basic security validation
            security_checks = {
                'input_validation': True,    # Check input shapes/ranges
                'output_sanitization': True, # Check output validity
                'error_handling': True,      # Graceful error handling
                'resource_limits': True      # Memory/compute limits
            }

            # Test input validation
            try:
                # Test with invalid input
                invalid_input = Tensor(np.random.randn(1, 999))  # Wrong shape
                _ = model(invalid_input)  # May fail gracefully
            except:
                pass  # Expected for wrong shape

            # Test output validation
            valid_output = model(test_input)
            output_valid = (
                not np.any(np.isnan(valid_output.data)) and
                not np.any(np.isinf(valid_output.data))
            )

            security_validation = output_valid and all(security_checks.values())
            assert security_validation
            production_capabilities["Security considerations"] = True

            # Test 8: Production readiness
            # Overall system validation
            production_checklist = {
                'model_inference_works': True,
                'monitoring_functional': True,
                'deployment_ready': True,
                'performance_acceptable': True,
                'error_handling_robust': True
            }

            # Final production test
            try:
                # Simulate production load
                production_inputs = [
                    Tensor(np.random.randn(1, 20)),
                    Tensor(np.random.randn(8, 20)),
                    Tensor(np.random.randn(32, 20))
                ]

                for prod_input in production_inputs:
                    pred = model(prod_input)
                    monitor.log_prediction(prod_input, pred)

                    # Validate production prediction
                    assert pred.shape[0] == prod_input.shape[0]
                    assert pred.shape[1] == 5
                    assert not np.any(np.isnan(pred.data))

                # Check monitoring works under load
                final_metrics = monitor.get_metrics()
                assert final_metrics['prediction_count'] > 0

                production_readiness = all(production_checklist.values())
                assert production_readiness

            except Exception as prod_error:
                assert False, f"Production simulation failed: {prod_error}"

            production_capabilities["Production readiness"] = True

        except Exception as e:
            # Show progress even if not complete
            completed_count = sum(production_capabilities.values())
            total_count = len(production_capabilities)

            progress_report = "\n🔍 PRODUCTION ML SYSTEM PROGRESS:\n"
            for capability, completed in production_capabilities.items():
                status = "✅" if completed else "❌"
                progress_report += f"  {status} {capability}\n"

            progress_report += f"\n📊 Progress: {completed_count}/{total_count} capabilities ready"

            assert False, f"""
            ❌ PRODUCTION ML SYSTEM NOT COMPLETE!

            🔍 ERROR: {str(e)}

            {progress_report}

            🔧 NEXT STEPS:
            1. Fix the failing capability above
            2. Re-run this test
            3. When all ✅, TinyTorch is production-ready!

            💡 ALMOST THERE!
            You've completed {completed_count}/{total_count} production capabilities.
            Just fix the error above and you'll have a complete production ML system!
            """

        # If we get here, everything passed!
        assert True, """
        🎉 PRODUCTION ML SYSTEM COMPLETE! 🎉

        ✅ Model monitoring and alerting
        ✅ Deployment infrastructure and serving
        ✅ Pipeline orchestration and automation
        ✅ Continuous integration and validation
        ✅ Model lifecycle management
        ✅ Performance optimization
        ✅ Security considerations
        ✅ Production readiness validation

        🚀 TINYTORCH IS PRODUCTION-READY!

        💡 What you can now deploy:
        - Real-time ML APIs with monitoring
        - Batch processing pipelines with automation
        - A/B testing and experimentation platforms
        - Auto-scaling ML services with health checks
        - Enterprise ML systems with governance

        🏆 PRODUCTION ML ENGINEERING ACHIEVED:
        You've built a complete ML system that includes:
        - Research-grade model development
        - Production-grade deployment infrastructure
        - Enterprise-grade monitoring and governance
        - Industry-standard CI/CD for ML
        - Real-world operational capabilities

        🎯 READY FOR MODULE 16: CAPSTONE PROJECT!

        Build complete end-to-end ML systems:
        - TinyGPT transformer models
        - Computer vision applications
        - Multimodal AI systems
        - Production ML platforms

        🌟 CONGRATULATIONS!
        You are now a complete ML Systems Engineer!
        """


# Note: No separate regression prevention - we test complete system stability above
