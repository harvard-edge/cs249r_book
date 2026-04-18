import Chart from 'chart.js/auto';

export class FlashcardStats {
    constructor(shadowRoot) {
        this.shadowRoot = shadowRoot;
        this.charts = {
            progress: null,
            quality: null
        };
        this.colors = {
            primary: '#3b82f6',      // Blue-500
            lighter: '#60a5fa',      // Blue-400
            darker: '#2563eb',       // Blue-600
            muted: '#93c5fd',        // Blue-300
            background: '#eff6ff'    // Blue-50
        };
    }

    async showStats(flashcards) {
        const statsView = this.shadowRoot.querySelector('#statsView');
        if (!statsView) return;

        // Get theme detection
        const hostElement = this.shadowRoot.host;
        const currentTheme = hostElement?.getAttribute('data-socratiq-theme') || 'light';
        const isDark = currentTheme === 'dark';

        // Clear any existing charts
        if (this.charts.progress) this.charts.progress.destroy();
        if (this.charts.quality) this.charts.quality.destroy();

        // Clear and setup stats container
        statsView.innerHTML = `
            <div class="p-4 space-y-4">
                <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div class="p-4 rounded-lg shadow" style="background-color: ${isDark ? '#0d1117' : '#ffffff'} !important; color: ${isDark ? '#e6edf3' : '#1f2328'} !important;">
                        <h3 class="text-sm font-medium" style="color: ${isDark ? '#9ca3af' : '#6b7280'} !important;">Total Cards</h3>
                        <p class="text-2xl font-bold text-blue-500">${flashcards.length}</p>
                    </div>
                    <div class="p-4 rounded-lg shadow" style="background-color: ${isDark ? '#0d1117' : '#ffffff'} !important; color: ${isDark ? '#e6edf3' : '#1f2328'} !important;">
                        <h3 class="text-sm font-medium" style="color: ${isDark ? '#9ca3af' : '#6b7280'} !important;">Mastered</h3>
                        <p class="text-2xl font-bold text-blue-500">${flashcards.filter(card => 
                            card.reviewHistory.length > 0 && 
                            card.reviewHistory[card.reviewHistory.length - 1].quality >= 4
                        ).length}</p>
                    </div>
                    <div class="p-4 rounded-lg shadow" style="background-color: ${isDark ? '#0d1117' : '#ffffff'} !important; color: ${isDark ? '#e6edf3' : '#1f2328'} !important;">
                        <h3 class="text-sm font-medium" style="color: ${isDark ? '#9ca3af' : '#6b7280'} !important;">Need Review</h3>
                        <p class="text-2xl font-bold text-blue-500">${flashcards.filter(card => 
                            card.nextReviewDate <= new Date()
                        ).length}</p>
                    </div>
                </div>
                
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div class="p-4 rounded-lg shadow" style="background-color: ${isDark ? '#0d1117' : '#ffffff'} !important; color: ${isDark ? '#e6edf3' : '#1f2328'} !important;">
                        <h3 class="text-sm font-medium mb-4" style="color: ${isDark ? '#9ca3af' : '#6b7280'} !important;">Learning Progress</h3>
                        <canvas id="progressChart"></canvas>
                    </div>
                    <div class="p-4 rounded-lg shadow" style="background-color: ${isDark ? '#0d1117' : '#ffffff'} !important; color: ${isDark ? '#e6edf3' : '#1f2328'} !important;">
                        <h3 class="text-sm font-medium mb-4" style="color: ${isDark ? '#9ca3af' : '#6b7280'} !important;">Quality Distribution</h3>
                        <canvas id="qualityChart"></canvas>
                    </div>
                </div>
            </div>
        `;

        // Update colors based on theme
        this.colors = {
            primary: '#3b82f6',      // Blue-500
            lighter: '#60a5fa',      // Blue-400
            darker: '#2563eb',       // Blue-600
            muted: '#93c5fd',        // Blue-300
            background: isDark ? '#1e3a8a' : '#eff6ff'    // Dark blue for dark theme, light blue for light theme
        };

        await this.initializeCharts(flashcards, isDark);
    }

    async initializeCharts(flashcards, isDark = false) {
        const progressCtx = this.shadowRoot.querySelector('#progressChart');
        const qualityCtx = this.shadowRoot.querySelector('#qualityChart');

        if (!progressCtx || !qualityCtx) return;

        // Progress over time chart
        const progressData = this.getProgressData(flashcards);
        this.charts.progress = new Chart(progressCtx, {
            type: 'line',
            data: {
                labels: progressData.labels,
                datasets: [{
                    label: 'Cards Mastered',
                    data: progressData.data,
                    borderColor: this.colors.primary,
                    backgroundColor: this.colors.background,
                    fill: true,
                    tension: 0.1,
                    animation: {
                        duration: 2000,
                        easing: 'easeOutQuart'
                    }
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            color: isDark ? '#e6edf3' : '#1f2328'
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            color: isDark ? '#9ca3af' : '#6b7280'
                        },
                        grid: {
                            color: isDark ? '#30363d' : '#e5e7eb'
                        }
                    },
                    y: {
                        beginAtZero: true,
                        ticks: {
                            stepSize: 1,
                            color: isDark ? '#9ca3af' : '#6b7280'
                        },
                        grid: {
                            color: isDark ? '#30363d' : '#e5e7eb'
                        }
                    }
                },
                animation: {
                    duration: 2000,
                    easing: 'easeInOutQuart'
                }
            }
        });

        // Quality distribution chart
        const qualityData = this.getQualityDistribution(flashcards);
        this.charts.quality = new Chart(qualityCtx, {
            type: 'bar',
            data: {
                labels: ['Complete Blackout', 'Incorrect', 'Difficult', 'Hesitant', 'Good', 'Perfect'],
                datasets: [{
                    label: 'Number of Reviews',
                    data: qualityData,
                    backgroundColor: [
                        this.colors.primary,
                        this.colors.lighter,
                        this.colors.muted,
                        this.colors.darker,
                        this.colors.primary,
                        this.colors.lighter
                    ],
                    animation: {
                        delay: (context) => context.dataIndex * 100
                    }
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            color: isDark ? '#e6edf3' : '#1f2328'
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            color: isDark ? '#9ca3af' : '#6b7280'
                        },
                        grid: {
                            color: isDark ? '#30363d' : '#e5e7eb'
                        }
                    },
                    y: {
                        beginAtZero: true,
                        ticks: {
                            stepSize: 1,
                            color: isDark ? '#9ca3af' : '#6b7280'
                        },
                        grid: {
                            color: isDark ? '#30363d' : '#e5e7eb'
                        }
                    }
                },
                animation: {
                    duration: 1500,
                    easing: 'easeInOutQuart'
                }
            }
        });
    }

    getProgressData(flashcards) {
        const dates = flashcards
            .flatMap(card => card.reviewHistory)
            .map(review => review.date)
            .sort((a, b) => a - b);

        const uniqueDates = [...new Set(dates.map(date => 
            new Date(date).toLocaleDateString()
        ))];

        const masteredByDate = uniqueDates.map(date => {
            return flashcards.filter(card => 
                card.reviewHistory.some(review => 
                    new Date(review.date).toLocaleDateString() === date &&
                    review.quality >= 4
                )
            ).length;
        });

        return {
            labels: uniqueDates,
            data: masteredByDate
        };
    }

    getQualityDistribution(flashcards) {
        const distribution = new Array(6).fill(0);
        
        flashcards.forEach(card => {
            card.reviewHistory.forEach(review => {
                distribution[review.quality]++;
            });
        });

        return distribution;
    }
}