import { showPopover } from '../../../libs/utils/utils.js';
import { AnkiConverter } from '../utils/anki-converter.js';



export class SpacedRepetitionInteractionHandler {
    constructor(modal, storageHandler, shadowRoot) {
        this.modal = modal;
        this.storageHandler = storageHandler;
        this.shadowRoot = shadowRoot;

        // Add validation
        if (!this.storageHandler) {
            console.error('Storage handler is required but was not provided');
            throw new Error('Storage handler is required');
        }

        // Log for debugging
        console.log('Storage handler initialized:', {
            hasExportMethod: !!this.storageHandler.exportToJSON,
            storageHandler: this.storageHandler
        });

        this.setupInteractionButtons();
    }

    setupInteractionButtons() {
        // Import button
        const importBtn = this.modal.shadowRoot.querySelector('button[title="Import"]');
        if (importBtn) {
            importBtn.addEventListener('click', () => this.handleImport());
        }

        // Download button
        const downloadBtn = this.modal.shadowRoot.querySelector('#sr-download-btn');
        if (downloadBtn) {
            downloadBtn.addEventListener('click', () => this.handleDownload());
        }

        // Copy button
        const copyBtn = this.modal.shadowRoot.querySelector('#copyButton');
        if (copyBtn) {
            copyBtn.addEventListener('click', () => this.handleCopy());
        }

        // Share button
        const shareBtn = this.modal.shadowRoot.querySelector('#sr-share-btn');
        if (shareBtn) {
            shareBtn.addEventListener('click', () => this.handleShare());
        }
    }

    async handleImport() {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.json,.csv,.txt';

        input.addEventListener('change', async (e) => {
            try {
                const file = e.target.files[0];
                const reader = new FileReader();

                reader.onload = async (event) => {
                    try {
                        let importedData;
                        
                        if (file.name.endsWith('.json')) {
                            // Handle JSON format
                            importedData = JSON.parse(event.target.result);
                        } else {
                            // Handle Anki format (CSV/TXT)
                            importedData = AnkiConverter.parseAnkiCSV(event.target.result);
                        }
                        
                        // Show loading state
                        const loadingId = this.modal.notificationHandler.showLoadingState('Importing flashcards...');

                        try {
                            await this.storageHandler.mergeImportedData(importedData);
                            
                            this.modal.notificationHandler.updateNotification(
                                loadingId,
                                'Flashcards imported successfully!',
                                'success'
                            );

                            window.dispatchEvent(new CustomEvent('sr-data-updated', {
                                detail: { type: 'import-complete' }
                            }));

                            showPopover(this.shadowRoot, 'Data imported successfully!', 'success');
                        } catch (importError) {
                            console.error('Error importing data:', importError);
                            this.modal.notificationHandler.updateNotification(
                                loadingId,
                                'Failed to import flashcards',
                                'error'
                            );
                            showPopover(this.shadowRoot, 'Error importing data', 'error');
                        }
                    } catch (parseError) {
                        console.error('Error parsing imported file:', parseError);
                        showPopover(
                            this.shadowRoot,
                            'Invalid file format. Please check the file structure.',
                            'error'
                        );
                    }
                };

                reader.readAsText(file);
            } catch (error) {
                console.error('Error reading file:', error);
                showPopover(this.shadowRoot, 'Error reading file', 'error');
            }
        });

        input.click();
    }

    async handleDownload() {
        try {
            console.log("Downloading data...");
            
            // Add download format options
            const format = await this.showFormatDialog();
            
            // Only proceed if a format was selected
            if (!format) {
                return; // Exit if dialog was closed without selection
            }

            const exportData = await this.storageHandler.exportToJSON();
            let downloadData;
            let filename;
            
            if (format === 'json') {
                downloadData = JSON.stringify(exportData, null, 2);
                filename = `flashcards_backup_${new Date().toISOString().split('T')[0]}.json`;
            } else {
                downloadData = AnkiConverter.convertToAnkiFormat(exportData);
                filename = `flashcards_anki_${new Date().toISOString().split('T')[0]}.txt`;
            }

            const blob = new Blob([downloadData], { 
                type: format === 'json' ? 'application/json' : 'text/plain' 
            });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            
            a.href = url;
            a.download = filename;
            a.style.display = 'none';
            this.modal.shadowRoot.appendChild(a);
            a.click();
            this.modal.shadowRoot.removeChild(a);

            URL.revokeObjectURL(url);
            showPopover(this.shadowRoot, 'Download complete!', 'success');
        } catch (error) {
            console.error('Error downloading data:', error);
            showPopover(this.shadowRoot, 'Error downloading data', 'error');
        }
    }

    async showFormatDialog() {
        return new Promise((resolve) => {
            // Create dialog within the modal's shadow root
            const dialog = document.createElement('div');
            dialog.className = 'format-dialog';
            dialog.innerHTML = `
                <div class="format-dialog-overlay"></div>
                <div class="format-dialog-content">
                    <button class="close-btn" aria-label="Close dialog">
                        <kbd>esc</kbd>
                    </button>
                    <h3 class="text-lg font-semibold mb-4">Choose Export Format</h3>
                    <button class="json-btn">
                        <span class="flex items-center gap-2">
                            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3m-1 4l-3 3m0 0l-3-3m3 3V4" />
                            </svg>
                            JSON (App Format)
                        </span>
                    </button>
                    <button class="anki-btn">
                        <span class="flex items-center gap-2">
                            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                            </svg>
                            Anki Format (TXT)
                        </span>
                    </button>
                </div>
            `;
            
            const style = document.createElement('style');
            style.textContent = `
                .format-dialog {
                    position: fixed;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    z-index: 9999;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                
                .format-dialog-overlay {
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    background-color: rgba(0, 0, 0, 0.5);
                    backdrop-filter: blur(2px);
                }
                
                .format-dialog-content {
                    position: relative;
                    background: white;
                    padding: 1.5rem;
                    border-radius: 0.75rem;
                    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
                    min-width: 300px;
                    text-align: center;
                    z-index: 10000;
                }

                .close-btn {
                    position: absolute;
                    top: 0.75rem;
                    right: 0.75rem;
                    padding: 0.25rem;
                    border: none;
                    background: transparent;
                    color: rgb(156 163 175);
                    cursor: pointer;
                    border-radius: 0.375rem;
                    transition: all 0.2s;
                }

                .close-btn:hover {
                    color: rgb(107 114 128);
                    background: rgb(243 244 246);
                }
                
                .format-dialog button:not(.close-btn) {
                    display: block;
                    width: 100%;
                    margin: 0.75rem 0;
                    padding: 0.75rem 1rem;
                    border: none;
                    border-radius: 0.5rem;
                    cursor: pointer;
                    font-size: 0.875rem;
                    font-weight: 500;
                    transition: all 0.2s;
                }
                
                .json-btn {
                    background: rgb(59 130 246);
                    color: white;
                }
                
                .json-btn:hover {
                    background: rgb(29 78 216);
                }
                
                .anki-btn {
                    background: rgb(34 197 94);
                    color: white;
                }
                
                .anki-btn:hover {
                    background: rgb(21 128 61);
                }

                @media (prefers-color-scheme: dark) {
                    .format-dialog-content {
                        background: rgb(24 24 27);
                        color: white;
                    }
                    
                    .close-btn {
                        color: rgb(156 163 175);
                    }

                    .close-btn:hover {
                        color: rgb(209 213 219);
                        background: rgb(55 65 81);
                    }

                    .json-btn {
                        background: rgb(29 78 216);
                    }
                    
                    .json-btn:hover {
                        background: rgb(30 64 175);
                    }
                    
                    .anki-btn {
                        background: rgb(21 128 61);
                    }
                    
                    .anki-btn:hover {
                        background: rgb(22 101 52);
                    }
                }
            `;
            
            // Append to modal's shadow root
            dialog.appendChild(style);
            this.modal.shadowRoot.appendChild(dialog);

            // Function to cleanup and close dialog
            const closeDialog = () => {
                if (dialog && dialog.parentNode === this.modal.shadowRoot) {
                    this.modal.shadowRoot.removeChild(dialog);
                }
            };

            // Handle button clicks
            dialog.querySelector('.json-btn').addEventListener('click', () => {
                closeDialog();
                resolve('json');
            });
            
            dialog.querySelector('.anki-btn').addEventListener('click', () => {
                closeDialog();
                resolve('anki');
            });

            // Close button click
            dialog.querySelector('.close-btn').addEventListener('click', () => {
                closeDialog();
                resolve(null);
            });

            // Close on overlay click
            dialog.querySelector('.format-dialog-overlay').addEventListener('click', () => {
                closeDialog();
                resolve(null);
            });

            // Close on escape key
            const handleEscape = (e) => {
                if (e.key === 'Escape') {
                    closeDialog();
                    document.removeEventListener('keydown', handleEscape);
                    resolve(null);
                }
            };
            document.addEventListener('keydown', handleEscape);
        });
    }

    async handleCopy() {
        try {
            const exportData = await this.storageHandler.exportToJSON();
            await navigator.clipboard.writeText(JSON.stringify(exportData, null, 2));

            // Update button UI
            const copyIcon = this.modal.shadowRoot.querySelector('.copy-icon');
            const checkIcon = this.modal.shadowRoot.querySelector('.check-icon');
            
            copyIcon.classList.add('hidden');
            checkIcon.classList.remove('hidden');
            
            setTimeout(() => {
                copyIcon.classList.remove('hidden');
                checkIcon.classList.add('hidden');
            }, 1000);

            showPopover(this.shadowRoot, 'Copied to clipboard!', 'success');
        } catch (error) {
            console.error('Error copying data:', error);
            showPopover(this.shadowRoot, 'Error copying data', 'error');
        }
    }

    async handleShare() {
        try {
            const exportData = await this.storageHandler.exportToJSON();
            const dataStr = JSON.stringify(exportData, null, 2);
            const encodedData = encodeURIComponent(dataStr);
            const mailtoLink = `mailto:?subject=Flashcards%20Data&body=${encodedData}`;
            
            // Create and click a temporary link
            const a = document.createElement('a');
            a.href = mailtoLink;
            a.style.display = 'none';
            this.modal.shadowRoot.appendChild(a);
            a.click();
            this.modal.shadowRoot.removeChild(a);
            
            showPopover(this.shadowRoot, 'Opening email client...', 'success');
        } catch (error) {
            console.error('Error sharing data:', error);
            showPopover(this.shadowRoot, 'Error sharing data', 'error');
        }
    }
}
