// app/static/js/main.js - Core JavaScript functionality for the application

// Initialize Socket.IO connection
const socket = io();

// Global variables
let currentUser = null;
let notifications = [];

// DOM Ready
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupGlobalEventListeners();
    initializeNotifications();
});

// Initialize the application
function initializeApp() {
    console.log('🚀 BiometricID Platform initialized');
    
    // Check if user is authenticated
    checkAuthenticationStatus();
    
    // Initialize tooltips
    initializeTooltips();
    
    // Setup form validation
    setupFormValidation();
    
    // Initialize dark mode toggle if exists
    initializeDarkMode();
}

// Check authentication status
function checkAuthenticationStatus() {
    const userDropdown = document.getElementById('navbarDropdown');
    if (userDropdown) {
        currentUser = {
            username: userDropdown.textContent.trim(),
            authenticated: true
        };
        console.log('✅ User authenticated:', currentUser.username);
    }
}

// Initialize Bootstrap tooltips
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// Setup global event listeners
function setupGlobalEventListeners() {
    // Handle logout confirmation
    const logoutLinks = document.querySelectorAll('a[href*="logout"]');
    logoutLinks.forEach(link => {
        link.addEventListener('click', handleLogout);
    });
    
    // Handle form submissions with loading states
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', handleFormSubmit);
    });
    
    // Handle AJAX requests
    setupAjaxDefaults();
}

// Handle logout with confirmation
function handleLogout(event) {
    event.preventDefault();
    
    showConfirmDialog(
        'Logout Confirmation',
        'Are you sure you want to logout?',
        'warning',
        () => {
            window.location.href = event.target.href;
        }
    );
}

// Handle form submissions with loading states
function handleFormSubmit(event) {
    const form = event.target;
    const submitButton = form.querySelector('button[type="submit"]');
    
    if (submitButton && !submitButton.disabled) {
        // Show loading state
        showButtonLoading(submitButton);
        
        // Auto-hide loading state after 10 seconds (fallback)
        setTimeout(() => {
            hideButtonLoading(submitButton);
        }, 10000);
    }
}

// Show loading state on button
function showButtonLoading(button) {
    button.disabled = true;
    button.dataset.originalText = button.innerHTML;
    button.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Loading...';
}

// Hide loading state on button
function hideButtonLoading(button) {
    button.disabled = false;
    if (button.dataset.originalText) {
        button.innerHTML = button.dataset.originalText;
        delete button.dataset.originalText;
    }
}

// Setup AJAX defaults
function setupAjaxDefaults() {
    // Add CSRF token to all AJAX requests
    const csrfToken = document.querySelector('meta[name="csrf-token"]');
    if (csrfToken) {
        fetch.defaults = {
            headers: {
                'X-CSRFToken': csrfToken.getAttribute('content')
            }
        };
    }
}

// Form validation setup
function setupFormValidation() {
    const forms = document.querySelectorAll('.needs-validation');
    
    forms.forEach(form => {
        form.addEventListener('submit', function(event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        });
    });
}

// Notification system
function initializeNotifications() {
    // Create notification container if it doesn't exist
    if (!document.getElementById('notification-container')) {
        const container = document.createElement('div');
        container.id = 'notification-container';
        container.className = 'position-fixed top-0 end-0 p-3';
        container.style.zIndex = '1055';
        document.body.appendChild(container);
    }
    
    // Listen for server notifications
    socket.on('notification', function(data) {
        showNotification(data.message, data.type || 'info');
    });
}

// Show notification
function showNotification(message, type = 'info', duration = 5000) {
    const container = document.getElementById('notification-container');
    const notificationId = 'notification-' + Date.now();
    
    const notification = document.createElement('div');
    notification.id = notificationId;
    notification.className = `alert alert-${type} alert-dismissible fade show`;
    notification.setAttribute('role', 'alert');
    notification.innerHTML = `
        <strong>${getNotificationIcon(type)}</strong> ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    container.appendChild(notification);
    
    // Auto-remove notification after duration
    setTimeout(() => {
        const notificationElement = document.getElementById(notificationId);
        if (notificationElement) {
            const alert = new bootstrap.Alert(notificationElement);
            alert.close();
        }
    }, duration);
}

// Get notification icon based on type
function getNotificationIcon(type) {
    const icons = {
        'success': '<i class="fas fa-check-circle me-2"></i>',
        'danger': '<i class="fas fa-exclamation-circle me-2"></i>',
        'warning': '<i class="fas fa-exclamation-triangle me-2"></i>',
        'info': '<i class="fas fa-info-circle me-2"></i>'
    };
    return icons[type] || icons['info'];
}

// Show confirmation dialog
function showConfirmDialog(title, message, type = 'primary', onConfirm = null, onCancel = null) {
    const modalId = 'confirm-modal-' + Date.now();
    const modal = document.createElement('div');
    modal.id = modalId;
    modal.className = 'modal fade';
    modal.setAttribute('tabindex', '-1');
    modal.innerHTML = `
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">${title}</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <p>${message}</p>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                    <button type="button" class="btn btn-${type}" id="confirm-btn">Confirm</button>
                </div>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    
    const bootstrapModal = new bootstrap.Modal(modal);
    bootstrapModal.show();
    
    // Handle confirm button
    document.getElementById('confirm-btn').addEventListener('click', function() {
        bootstrapModal.hide();
        if (onConfirm) onConfirm();
    });
    
    // Handle cancel
    modal.addEventListener('hidden.bs.modal', function() {
        document.body.removeChild(modal);
        if (onCancel) onCancel();
    });
}

// Dark mode functionality
function initializeDarkMode() {
    const darkModeToggle = document.getElementById('darkModeToggle');
    if (darkModeToggle) {
        // Check for saved dark mode preference
        const isDarkMode = localStorage.getItem('darkMode') === 'true';
        if (isDarkMode) {
            document.body.classList.add('dark-mode');
        }
        
        darkModeToggle.addEventListener('click', function() {
            document.body.classList.toggle('dark-mode');
            const isDark = document.body.classList.contains('dark-mode');
            localStorage.setItem('darkMode', isDark);
        });
    }
}

// Utility functions
const Utils = {
    // Format timestamp
    formatTimestamp: function(timestamp) {
        const date = new Date(timestamp);
        return date.toLocaleString();
    },
    
    // Format file size
    formatFileSize: function(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },
    
    // Validate email
    validateEmail: function(email) {
        const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return re.test(email);
    },
    
    // Generate random ID
    generateId: function() {
        return Date.now().toString(36) + Math.random().toString(36).substr(2);
    },
    
    // Debounce function
    debounce: function(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },
    
    // Copy to clipboard
    copyToClipboard: function(text) {
        navigator.clipboard.writeText(text).then(function() {
            showNotification('Copied to clipboard!', 'success', 2000);
        }).catch(function() {
            showNotification('Failed to copy to clipboard', 'danger', 3000);
        });
    },
    
    // Show loading overlay
    showLoadingOverlay: function(message = 'Loading...') {
        const overlay = document.createElement('div');
        overlay.id = 'loading-overlay';
        overlay.className = 'position-fixed top-0 start-0 w-100 h-100 d-flex align-items-center justify-content-center';
        overlay.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
        overlay.style.zIndex = '9999';
        overlay.innerHTML = `
            <div class="text-center text-white">
                <div class="spinner-border mb-3" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <div>${message}</div>
            </div>
        `;
        document.body.appendChild(overlay);
    },
    
    // Hide loading overlay
    hideLoadingOverlay: function() {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.remove();
        }
    }
};

// Socket.IO event handlers
socket.on('connect', function() {
    console.log('🔌 Connected to server');
});

socket.on('disconnect', function() {
    console.log('🔌 Disconnected from server');
    showNotification('Connection lost. Please refresh the page.', 'warning');
});

socket.on('error', function(error) {
    console.error('Socket error:', error);
    showNotification('Connection error occurred', 'danger');
});

// Export for global use
window.BiometricApp = {
    showNotification,
    showConfirmDialog,
    Utils,
    socket
};

// Handle page visibility changes
document.addEventListener('visibilitychange', function() {
    if (document.hidden) {
        console.log('Page hidden');
    } else {
        console.log('Page visible');
    }
});

// Handle online/offline status
window.addEventListener('online', function() {
    showNotification('Connection restored', 'success', 3000);
});

window.addEventListener('offline', function() {
    showNotification('Connection lost - you are offline', 'warning');
});

console.log('✅ Main.js loaded successfully');