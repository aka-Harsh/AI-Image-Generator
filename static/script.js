// Global state management
window.AppState = {
    currentTheme: localStorage.getItem('theme') || 'light',
    isGenerating: false,
    generationProgress: 0,
    lastGeneration: null
};

// Theme Management
class ThemeManager {
    constructor() {
        this.init();
    }

    init() {
        this.applyTheme(window.AppState.currentTheme);
        this.createThemeToggle();
    }

    applyTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        window.AppState.currentTheme = theme;
        localStorage.setItem('theme', theme);
        
        // Update theme toggle icon
        const toggleIcon = document.querySelector('.theme-toggle-icon');
        if (toggleIcon) {
            toggleIcon.textContent = theme === 'dark' ? '☀️' : '🌙';
        }
    }

    createThemeToggle() {
        // Create theme toggle button
        const toggle = document.createElement('button');
        toggle.className = 'theme-toggle';
        toggle.innerHTML = `<span class="theme-toggle-icon">${window.AppState.currentTheme === 'dark' ? '☀️' : '🌙'}</span>`;
        toggle.title = 'Toggle theme';
        
        toggle.addEventListener('click', () => {
            const newTheme = window.AppState.currentTheme === 'light' ? 'dark' : 'light';
            this.applyTheme(newTheme);
        });
        
        document.body.appendChild(toggle);
    }

    toggle() {
        const newTheme = window.AppState.currentTheme === 'light' ? 'dark' : 'light';
        this.applyTheme(newTheme);
    }
}

// Drag and Drop Handler
class DragDropHandler {
    constructor() {
        this.init();
    }

    init() {
        this.setupDropZones();
    }

    setupDropZones() {
        // Find image input components in Gradio
        const imageInputs = document.querySelectorAll('input[type="file"][accept*="image"]');
        
        imageInputs.forEach(input => {
            const container = input.closest('.gradio-component');
            if (container) {
                this.makeDroppable(container, input);
            }
        });
    }

    makeDroppable(container, input) {
        container.addEventListener('dragover', (e) => {
            e.preventDefault();
            container.classList.add('drag-over');
        });

        container.addEventListener('dragleave', (e) => {
            e.preventDefault();
            container.classList.remove('drag-over');
        });

        container.addEventListener('drop', (e) => {
            e.preventDefault();
            container.classList.remove('drag-over');
            
            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].type.startsWith('image/')) {
                // Trigger file input
                input.files = files;
                input.dispatchEvent(new Event('change', { bubbles: true }));
                this.showDropSuccess(container);
            }
        });

        // Add visual indicator
        this.addDropIndicator(container);
    }

    addDropIndicator(container) {
        const indicator = document.createElement('div');
        indicator.className = 'drop-indicator';
        indicator.innerHTML = '📁 Drop image here or click to upload';
        indicator.style.cssText = `
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(79, 70, 229, 0.1);
            border: 2px dashed var(--primary-color);
            border-radius: var(--border-radius);
            padding: 2rem;
            text-align: center;
            color: var(--primary-color);
            font-weight: 500;
            opacity: 0;
            transition: opacity 0.2s ease;
            pointer-events: none;
            z-index: 10;
        `;
        
        container.style.position = 'relative';
        container.appendChild(indicator);

        // Show indicator on drag over
        container.addEventListener('dragover', () => {
            indicator.style.opacity = '1';
        });

        container.addEventListener('dragleave', () => {
            indicator.style.opacity = '0';
        });

        container.addEventListener('drop', () => {
            indicator.style.opacity = '0';
        });
    }

    showDropSuccess(container) {
        const success = document.createElement('div');
        success.className = 'drop-success';
        success.innerHTML = '✅ Image uploaded successfully!';
        success.style.cssText = `
            position: absolute;
            top: 10px;
            right: 10px;
            background: var(--success-color);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: var(--border-radius);
            font-size: 0.875rem;
            font-weight: 500;
            z-index: 20;
            animation: slideUp 0.3s ease-out;
        `;

        container.appendChild(success);
        
        setTimeout(() => {
            success.remove();
        }, 3000);
    }
}

// Progress Tracking
class ProgressTracker {
    constructor() {
        this.progressBar = null;
        this.init();
    }

    init() {
        this.createProgressBar();
    }

    createProgressBar() {
        const progressContainer = document.createElement('div');
        progressContainer.id = 'generation-progress';
        progressContainer.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            z-index: 1000;
            background: var(--surface-color);
            border-bottom: 1px solid var(--border-color);
            padding: 1rem;
            transform: translateY(-100%);
            transition: transform 0.3s ease;
        `;

        progressContainer.innerHTML = `
            <div class="progress-info">
                <span class="progress-text">Generating image...</span>
                <span class="progress-percentage">0%</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: 0%"></div>
            </div>
        `;

        document.body.appendChild(progressContainer);
        this.progressBar = progressContainer;
    }

    show() {
        if (this.progressBar) {
            this.progressBar.style.transform = 'translateY(0)';
            window.AppState.isGenerating = true;
        }
    }

    hide() {
        if (this.progressBar) {
            this.progressBar.style.transform = 'translateY(-100%)';
            window.AppState.isGenerating = false;
        }
    }

    update(percentage, text = 'Generating image...') {
        if (this.progressBar) {
            const fill = this.progressBar.querySelector('.progress-fill');
            const textEl = this.progressBar.querySelector('.progress-text');
            const percentEl = this.progressBar.querySelector('.progress-percentage');
            
            if (fill) fill.style.width = `${percentage}%`;
            if (textEl) textEl.textContent = text;
            if (percentEl) percentEl.textContent = `${Math.round(percentage)}%`;
        }
    }
}

// Notification System
class NotificationManager {
    constructor() {
        this.container = null;
        this.init();
    }

    init() {
        this.createContainer();
    }

    createContainer() {
        const container = document.createElement('div');
        container.id = 'notifications';
        container.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1000;
            max-width: 400px;
        `;
        document.body.appendChild(container);
        this.container = container;
    }

    show(message, type = 'info', duration = 5000) {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        
        const icons = {
            success: '✅',
            error: '❌',
            warning: '⚠️',
            info: 'ℹ️'
        };

        notification.innerHTML = `
            <div class="notification-content">
                <span class="notification-icon">${icons[type]}</span>
                <span class="notification-message">${message}</span>
                <button class="notification-close">×</button>
            </div>
        `;

        notification.style.cssText = `
            background: var(--surface-color);
            border: 1px solid var(--border-color);
            border-radius: var(--border-radius);
            box-shadow: var(--shadow-lg);
            margin-bottom: 0.5rem;
            animation: slideIn 0.3s ease-out;
            overflow: hidden;
        `;

        const content = notification.querySelector('.notification-content');
        content.style.cssText = `
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 1rem;
        `;

        const closeBtn = notification.querySelector('.notification-close');
        closeBtn.style.cssText = `
            background: none;
            border: none;
            font-size: 1.25rem;
            cursor: pointer;
            color: var(--text-secondary);
            margin-left: auto;
        `;

        closeBtn.addEventListener('click', () => {
            this.remove(notification);
        });

        this.container.appendChild(notification);

        if (duration > 0) {
            setTimeout(() => {
                this.remove(notification);
            }, duration);
        }

        return notification;
    }

    remove(notification) {
        notification.style.animation = 'slideOut 0.3s ease-in forwards';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }

    success(message, duration) {
        return this.show(message, 'success', duration);
    }

    error(message, duration) {
        return this.show(message, 'error', duration);
    }

    warning(message, duration) {
        return this.show(message, 'warning', duration);
    }

    info(message, duration) {
        return this.show(message, 'info', duration);
    }
}

// Image Gallery Enhancements
class GalleryManager {
    constructor() {
        this.init();
    }

    init() {
        this.enhanceGallery();
        this.setupImageViewer();
    }

    enhanceGallery() {
        // Find Gradio gallery components
        const galleries = document.querySelectorAll('.gradio-gallery');
        galleries.forEach(gallery => {
            this.addGalleryFeatures(gallery);
        });
    }

    addGalleryFeatures(gallery) {
        // Add fullscreen view capability
        const images = gallery.querySelectorAll('img');
        images.forEach(img => {
            img.style.cursor = 'pointer';
            img.addEventListener('click', () => {
                this.openImageViewer(img.src);
            });
        });
    }

    setupImageViewer() {
        // Create image viewer modal
        const viewer = document.createElement('div');
        viewer.id = 'image-viewer';
        viewer.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.9);
            z-index: 2000;
            display: flex;
            align-items: center;
            justify-content: center;
            opacity: 0;
            visibility: hidden;
            transition: all 0.3s ease;
        `;

        viewer.innerHTML = `
            <img id="viewer-image" style="max-width: 90%; max-height: 90%; object-fit: contain;">
            <button id="viewer-close" style="
                position: absolute;
                top: 20px;
                right: 20px;
                background: rgba(255, 255, 255, 0.1);
                border: none;
                color: white;
                font-size: 2rem;
                width: 50px;
                height: 50px;
                border-radius: 50%;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
            ">×</button>
        `;

        document.body.appendChild(viewer);

        // Close viewer events
        const closeBtn = viewer.querySelector('#viewer-close');
        closeBtn.addEventListener('click', () => this.closeImageViewer());
        
        viewer.addEventListener('click', (e) => {
            if (e.target === viewer) {
                this.closeImageViewer();
            }
        });

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeImageViewer();
            }
        });

        this.viewer = viewer;
    }

    openImageViewer(src) {
        const img = this.viewer.querySelector('#viewer-image');
        img.src = src;
        this.viewer.style.opacity = '1';
        this.viewer.style.visibility = 'visible';
    }

    closeImageViewer() {
        this.viewer.style.opacity = '0';
        this.viewer.style.visibility = 'hidden';
    }
}

// Keyboard Shortcuts
class KeyboardManager {
    constructor() {
        this.init();
    }

    init() {
        this.setupShortcuts();
    }

    setupShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + Enter = Generate
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                e.preventDefault();
                this.triggerGeneration();
            }
            
            // Ctrl/Cmd + T = Toggle theme
            if ((e.ctrlKey || e.metaKey) && e.key === 't') {
                e.preventDefault();
                if (window.themeManager) {
                    window.themeManager.toggle();
                }
            }
            
            // Escape = Cancel/Close
            if (e.key === 'Escape') {
                // Handled by individual components
            }
        });
    }

    triggerGeneration() {
        // Find and click the generate button
        const generateBtn = document.querySelector('button[variant="primary"]');
        if (generateBtn && !window.AppState.isGenerating) {
            generateBtn.click();
            window.notificationManager?.info('Generation started! (Ctrl+Enter)', 2000);
        }
    }
}

// Auto-save functionality
class AutoSaveManager {
    constructor() {
        this.saveKey = 'aiGenerator_autoSave';
        this.init();
    }

    init() {
        this.loadSavedData();
        this.setupAutoSave();
    }

    setupAutoSave() {
        // Auto-save form data every 30 seconds
        setInterval(() => {
            this.saveFormData();
        }, 30000);

        // Save on page unload
        window.addEventListener('beforeunload', () => {
            this.saveFormData();
        });
    }

    saveFormData() {
        try {
            const formData = {
                prompt: this.getFieldValue('prompt'),
                negativePrompt: this.getFieldValue('negative-prompt'),
                width: this.getFieldValue('width'),
                height: this.getFieldValue('height'),
                steps: this.getFieldValue('steps'),
                cfgScale: this.getFieldValue('cfg-scale'),
                seed: this.getFieldValue('seed'),
                timestamp: Date.now()
            };

            localStorage.setItem(this.saveKey, JSON.stringify(formData));
        } catch (error) {
            console.warn('Auto-save failed:', error);
        }
    }

    loadSavedData() {
        try {
            const saved = localStorage.getItem(this.saveKey);
            if (saved) {
                const data = JSON.parse(saved);
                
                // Only restore if saved within last 24 hours
                if (Date.now() - data.timestamp < 24 * 60 * 60 * 1000) {
                    setTimeout(() => {
                        this.restoreFormData(data);
                        window.notificationManager?.info('Previous session restored', 3000);
                    }, 1000);
                }
            }
        } catch (error) {
            console.warn('Auto-restore failed:', error);
        }
    }

    getFieldValue(fieldName) {
        const field = document.querySelector(`[data-testid="${fieldName}"] input, [data-testid="${fieldName}"] textarea`);
        return field ? field.value : '';
    }

    setFieldValue(fieldName, value) {
        const field = document.querySelector(`[data-testid="${fieldName}"] input, [data-testid="${fieldName}"] textarea`);
        if (field && value) {
            field.value = value;
            field.dispatchEvent(new Event('input', { bubbles: true }));
        }
    }

    restoreFormData(data) {
        if (data.prompt) this.setFieldValue('prompt', data.prompt);
        if (data.negativePrompt) this.setFieldValue('negative-prompt', data.negativePrompt);
        // Add more field restorations as needed
    }
}

// Add custom CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
    
    .drag-over {
        border-color: var(--primary-color) !important;
        background-color: rgba(79, 70, 229, 0.05) !important;
    }
    
    .notification {
        margin-bottom: 0.5rem;
    }
`;
document.head.appendChild(style);

// Initialize all managers when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Initialize managers
    window.themeManager = new ThemeManager();
    window.dragDropHandler = new DragDropHandler();
    window.progressTracker = new ProgressTracker();
    window.notificationManager = new NotificationManager();
    window.galleryManager = new GalleryManager();
    window.keyboardManager = new KeyboardManager();
    window.autoSaveManager = new AutoSaveManager();
    
    console.log('🎨 Enhanced AI Image Generator - Client-side initialized!');
    
    // Show welcome message
    setTimeout(() => {
        window.notificationManager?.success('Enhanced UI loaded! Press Ctrl+Enter to generate, Ctrl+T to toggle theme.', 5000);
    }, 1000);
});

// Handle Gradio updates (when components re-render)
const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
        if (mutation.addedNodes.length > 0) {
            // Re-initialize drag-drop for new image inputs
            if (window.dragDropHandler) {
                window.dragDropHandler.setupDropZones();
            }
            
            // Re-enhance galleries
            if (window.galleryManager) {
                window.galleryManager.enhanceGallery();
            }
        }
    });
});

observer.observe(document.body, {
    childList: true,
    subtree: true
});

// Export for global access
window.EnhancedAI = {
    ThemeManager,
    DragDropHandler,
    ProgressTracker,
    NotificationManager,
    GalleryManager,
    KeyboardManager,
    AutoSaveManager
};