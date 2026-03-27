// app/static/js/liveness.js - Liveness Detection JavaScript

class LivenessVerification {
    constructor() {
        this.socket = io();
        this.localVideo = null;
        this.remoteVideo = null;
        this.mediaStream = null;
        this.isVerificationActive = false;
        this.sessionId = null;
        this.currentChallenge = null;
        this.verificationResult = null;
        
        // Configuration
        this.config = {
            video: {
                width: 640,
                height: 480,
                facingMode: 'user'
            },
            frameRate: 15, // Send frame every 15 FPS
            quality: 0.8   // JPEG quality
        };
        
        this.initializeElements();
        this.setupEventListeners();
    }
    
    initializeElements() {
        this.localVideo = document.getElementById('localVideo');
        this.remoteVideo = document.getElementById('remoteVideo');
        this.startButton = document.getElementById('startVerification');
        this.stopButton = document.getElementById('stopVerification');
        this.statusDiv = document.getElementById('verificationStatus');
        this.messageDiv = document.getElementById('verificationMessage');
        this.challengeDiv = document.getElementById('currentChallenge');
        this.aiStatusDiv = document.getElementById('aiStatus');
        this.resultDiv = document.getElementById('verificationResult');
        this.historyDiv = document.getElementById('verificationHistory');
        
        // Progress elements
        this.progressBar = document.getElementById('verificationProgress');
        this.progressText = document.getElementById('progressText');
        
        console.log('✅ Liveness verification elements initialized');
    }
    
    setupEventListeners() {
        // Button event listeners
        if (this.startButton) {
            this.startButton.addEventListener('click', () => this.startVerification());
        }
        
        if (this.stopButton) {
            this.stopButton.addEventListener('click', () => this.stopVerification());
        }
        
        // Socket event listeners
        this.socket.on('verification_started', (data) => this.handleVerificationStarted(data));
        this.socket.on('frame_processed', (data) => this.handleFrameProcessed(data));
        this.socket.on('verification_complete', (data) => this.handleVerificationComplete(data));
        this.socket.on('verification_stopped', () => this.handleVerificationStopped());
        this.socket.on('error', (data) => this.handleError(data));
        
        // Window events
        window.addEventListener('beforeunload', () => this.cleanup());
        
        console.log('✅ Event listeners setup complete');
    }
    
    async startVerification() {
        try {
            this.updateStatus('Initializing camera...', 'info');
            this.showButtonLoading(this.startButton);
            
            // Request camera access
            await this.initializeCamera();
            
            // Start verification session
            this.socket.emit('start_verification');
            
            this.updateStatus('Starting verification...', 'info');
            
        } catch (error) {
            console.error('Error starting verification:', error);
            this.handleError({ message: 'Failed to start verification: ' + error.message });
            this.hideButtonLoading(this.startButton);
        }
    }
    
    async initializeCamera() {
        try {
            // Stop any existing stream
            if (this.mediaStream) {
                this.mediaStream.getTracks().forEach(track => track.stop());
            }
            
            // Request camera access
            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: this.config.video.width,
                    height: this.config.video.height,
                    facingMode: this.config.video.facingMode
                },
                audio: false
            });
            
            // Set video stream
            this.localVideo.srcObject = this.mediaStream;
            this.localVideo.play();
            
            console.log('✅ Camera initialized successfully');
            
        } catch (error) {
            console.error('Camera initialization error:', error);
            throw new Error('Camera access denied or unavailable');
        }
    }
    
    handleVerificationStarted(data) {
        this.sessionId = data.session_id;
        this.currentChallenge = data.challenge;
        this.isVerificationActive = true;
        
        this.updateStatus('Verification started', 'success');
        this.updateChallenge(this.currentChallenge);
        
        // Update UI
        this.hideButtonLoading(this.startButton);
        this.startButton.style.display = 'none';
        this.stopButton.style.display = 'inline-block';
        this.resultDiv.style.display = 'none';
        
        // Start sending frames
        this.startFrameCapture();
        
        console.log('✅ Verification session started:', data);
    }
    
    startFrameCapture() {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        canvas.width = this.config.video.width;
        canvas.height = this.config.video.height;
        
        const captureFrame = () => {
            if (!this.isVerificationActive) return;
            
            // Draw current video frame to canvas
            ctx.drawImage(this.localVideo, 0, 0, canvas.width, canvas.height);
            
            // Convert to base64
            const dataURL = canvas.toDataURL('image/jpeg', this.config.quality);
            
            // Send frame to server
            this.socket.emit('process_frame', {
                frame: dataURL,
                session_id: this.sessionId
            });
            
            // Schedule next frame
            setTimeout(captureFrame, 1000 / this.config.frameRate);
        };
        
        // Start capturing frames
        captureFrame();
    }
    
    handleFrameProcessed(data) {
        // Update status and message
        this.updateStatus(data.message, this.getStatusType(data.status));
        this.updateAIStatus(data.ai_status);
        
        // Update challenge if changed
        if (data.challenge !== this.currentChallenge) {
            this.currentChallenge = data.challenge;
            this.updateChallenge(this.currentChallenge);
        }
        
        // Display processed frame
        if (data.processed_frame && this.remoteVideo) {
            this.remoteVideo.src = data.processed_frame;
        }
        
        // Update progress
        this.updateProgress(data.status);
        
        // Handle completion
        if (data.is_complete) {
            this.isVerificationActive = false;
        }
    }
    
    handleVerificationComplete(data) {
        this.verificationResult = data.result;
        this.isVerificationActive = false;
        
        // Update UI
        this.updateStatus(data.message, this.getStatusType(data.result));
        this.showResult(data.result, data.message);
        
        // Reset buttons
        this.startButton.style.display = 'inline-block';
        this.stopButton.style.display = 'none';
        this.hideButtonLoading(this.startButton);
        
        // Stop camera
        this.cleanup();
        
        // Reload verification history
        this.loadVerificationHistory();
        
        console.log('✅ Verification completed:', data);
    }
    
    stopVerification() {
        this.isVerificationActive = false;
        this.socket.emit('stop_verification');
        
        this.updateStatus('Stopping verification...', 'warning');
        this.cleanup();
    }
    
    handleVerificationStopped() {
        this.isVerificationActive = false;
        this.updateStatus('Verification stopped', 'warning');
        
        // Reset UI
        this.startButton.style.display = 'inline-block';
        this.stopButton.style.display = 'none';
        this.hideButtonLoading(this.startButton);
        
        console.log('✅ Verification stopped');
    }
    
    handleError(data) {
        console.error('Verification error:', data);
        this.updateStatus(data.message, 'danger');
        this.isVerificationActive = false;
        
        // Reset UI
        this.startButton.style.display = 'inline-block';
        this.stopButton.style.display = 'none';
        this.hideButtonLoading(this.startButton);
        
        this.cleanup();
    }
    
    updateStatus(message, type = 'info') {
        if (this.statusDiv) {
            this.statusDiv.className = `alert alert-${type}`;
            this.statusDiv.textContent = message;
        }
        
        if (this.messageDiv) {
            this.messageDiv.textContent = message;
        }
    }
    
    updateChallenge(challenge) {
        if (this.challengeDiv) {
            const challengeText = challenge.replace('_', ' ').toUpperCase();
            this.challengeDiv.innerHTML = `
                <strong>Current Challenge:</strong> 
                <span class="badge bg-primary">${challengeText}</span>
            `;
        }
    }
    
    updateAIStatus(status) {
        if (this.aiStatusDiv) {
            const isReal = status.includes('REAL');
            const badgeClass = isReal ? 'bg-success' : 'bg-danger';
            this.aiStatusDiv.innerHTML = `
                <span class="badge ${badgeClass}">${status}</span>
            `;
        }
    }
    
    updateProgress(status) {
        if (!this.progressBar || !this.progressText) return;
        
        const progressMap = {
            'processing': { value: 20, text: 'Processing...' },
            'calibrating': { value: 40, text: 'Calibrating...' },
            'challenge': { value: 60, text: 'Performing challenge...' },
            'verifying': { value: 80, text: 'Verifying...' },
            'success': { value: 100, text: 'Verification successful!' },
            'verified': { value: 100, text: 'Verification complete!' },
            'failed': { value: 100, text: 'Verification failed' },
            'spoofing': { value: 100, text: 'Spoofing detected' }
        };
        
        const progress = progressMap[status] || { value: 0, text: 'Initializing...' };
        
        this.progressBar.style.width = `${progress.value}%`;
        this.progressBar.setAttribute('aria-valuenow', progress.value);
        this.progressText.textContent = progress.text;
        
        // Update progress bar color based on status
        this.progressBar.className = 'progress-bar';
        if (status === 'failed' || status === 'spoofing') {
            this.progressBar.classList.add('bg-danger');
        } else if (status === 'success' || status === 'verified') {
            this.progressBar.classList.add('bg-success');
        } else {
            this.progressBar.classList.add('bg-primary');
        }
    }
    
    showResult(result, message) {
        if (!this.resultDiv) return;
        
        const isSuccess = result === 'verified';
        const iconClass = isSuccess ? 'fa-check-circle text-success' : 'fa-times-circle text-danger';
        const titleClass = isSuccess ? 'text-success' : 'text-danger';
        
        this.resultDiv.innerHTML = `
            <div class="card">
                <div class="card-body text-center">
                    <i class="fas ${iconClass} fa-3x mb-3"></i>
                    <h4 class="${titleClass}">${isSuccess ? 'Verification Successful' : 'Verification Failed'}</h4>
                    <p class="text-muted">${message}</p>
                    ${isSuccess ? 
                        '<p class="text-success"><i class="fas fa-shield-alt"></i> Identity verified successfully</p>' : 
                        '<p class="text-danger"><i class="fas fa-exclamation-triangle"></i> Please try again</p>'
                    }
                </div>
            </div>
        `;
        
        this.resultDiv.style.display = 'block';
    }
    
    getStatusType(status) {
        const statusMap = {
            'processing': 'info',
            'success': 'success',
            'verified': 'success',
            'failed': 'danger',
            'spoofing': 'danger',
            'error': 'danger'
        };
        return statusMap[status] || 'info';
    }
    
    showButtonLoading(button) {
        if (button) {
            button.disabled = true;
            button.dataset.originalText = button.innerHTML;
            button.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Loading...';
        }
    }
    
    hideButtonLoading(button) {
        if (button) {
            button.disabled = false;
            if (button.dataset.originalText) {
                button.innerHTML = button.dataset.originalText;
                delete button.dataset.originalText;
            }
        }
    }
    
    async loadVerificationHistory() {
        try {
            const response = await fetch('/liveness/api/verification_history');
            if (response.ok) {
                const history = await response.json();
                this.displayVerificationHistory(history);
            }
        } catch (error) {
            console.error('Error loading verification history:', error);
        }
    }
    
    displayVerificationHistory(history) {
        if (!this.historyDiv || !history.length) return;
        
        const historyHtml = history.map(item => `
            <div class="card mb-2">
                <div class="card-body p-3">
                    <div class="row align-items-center">
                        <div class="col-md-3">
                            <strong>${item.challenge_type.replace('_', ' ').toUpperCase()}</strong>
                        </div>
                        <div class="col-md-3">
                            <span class="badge ${item.result === 'verified' ? 'bg-success' : 'bg-danger'}">
                                ${item.result.toUpperCase()}
                            </span>
                        </div>
                        <div class="col-md-3">
                            <small class="text-muted">
                                Confidence: ${(item.ai_confidence * 100).toFixed(1)}%
                            </small>
                        </div>
                        <div class="col-md-3">
                            <small class="text-muted">
                                ${new Date(item.timestamp).toLocaleString()}
                            </small>
                        </div>
                    </div>
                </div>
            </div>
        `).join('');
        
        this.historyDiv.innerHTML = `
            <h6>Recent Verification History</h6>
            ${historyHtml}
        `;
    }
    
    cleanup() {
        // Stop media stream
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }
        
        // Clear video sources
        if (this.localVideo) {
            this.localVideo.srcObject = null;
        }
        
        if (this.remoteVideo) {
            this.remoteVideo.src = '';
        }
        
        console.log('✅ Cleanup completed');
    }
    
    // Check browser compatibility
    static checkBrowserSupport() {
        const requirements = {
            getUserMedia: navigator.mediaDevices && navigator.mediaDevices.getUserMedia,
            canvas: !!document.createElement('canvas').getContext('2d'),
            socketIO: typeof io !== 'undefined',
            webRTC: !!(window.RTCPeerConnection || window.webkitRTCPeerConnection || window.mozRTCPeerConnection)
        };
        
        const unsupported = Object.keys(requirements).filter(key => !requirements[key]);
        
        if (unsupported.length > 0) {
            console.warn('Browser compatibility issues:', unsupported);
            return false;
        }
        
        return true;
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    // Check if we're on the verification page
    if (document.getElementById('localVideo')) {
        // Check browser support
        if (!LivenessVerification.checkBrowserSupport()) {
            document.getElementById('verificationStatus').innerHTML = `
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-triangle"></i>
                    Your browser doesn't support all required features for liveness verification.
                    Please use a modern browser like Chrome, Firefox, or Safari.
                </div>
            `;
            return;
        }
        
        // Initialize liveness verification
        window.livenessVerification = new LivenessVerification();
        
        // Load initial verification history
        window.livenessVerification.loadVerificationHistory();
        
        console.log('✅ Liveness verification initialized');
    }
});

// Export for global use
window.LivenessVerification = LivenessVerification;