// src_shadow\js\components\quiz\showQuizStats.js
import {STORAGE_KEY_CHAPTERS} from '../../../configs/env_configs.js'
import {getChapterProgress} from './quiz-storage.js'
import {getDashboardChapterData} from './chapterMapService.js'
import {verifyPDFReport} from './verifyReport.js'
import {
    // getLastVisitedChapter,
    parseNavigation  // Add this import
} from './navParser.js'

import {
    Chart,
    DoughnutController,
    LineController,
    BarController,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    BarElement,
    ArcElement,
    Legend,
    Tooltip
} from 'chart.js';
import { getQuizStats } from './quiz-storage.js';
import { generateQuizPDF } from './generateQuizPDF.js';
import { enableTooltip } from '../tooltip/tooltip.js';
import { showPopover } from '../../libs/utils/utils.js';


// Register the chart components we need
Chart.register(
    DoughnutController,
    LineController,
    BarController,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    BarElement,
    ArcElement,
    Legend,
    Tooltip
);

let chartInstances = {
    progressChart: null,
    successRateChart: null,
    dailyQuizzesChart: null,
    performanceTrendChart: null
};


function showVerificationModal(shadowRoot, verificationResult, fileName) {
    // Create verification modal
    const verificationModal = document.createElement('div');
    verificationModal.className = 'verification-modal fixed inset-0 z-60 overflow-y-auto';
    
    const statusColor = verificationResult.isValid ? 'green' : 'red';
    const statusIcon = verificationResult.isValid ? 
        '<svg class="w-16 h-16 text-green-500 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>' :
        '<svg class="w-16 h-16 text-red-500 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>';
    
    const verificationHTML = `
        <div class="overlay fixed inset-0 bg-black bg-opacity-60"></div>
        <div class="min-h-screen px-4 text-center">
            <span class="inline-block h-screen align-middle" aria-hidden="true">&#8203;</span>
            
            <div class="verification-modal-content inline-block w-full max-w-md my-8 text-left align-middle transition-all transform bg-white rounded-lg shadow-xl">
                <!-- Header -->
                <div class="px-6 py-4 border-b border-gray-200 rounded-t-lg">
                    <div class="flex justify-between items-center">
                        <h3 class="text-lg font-semibold text-gray-900">PDF Verification Result</h3>
                        <button class="close-verification-modal text-gray-500 hover:text-gray-700">
                            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                            </svg>
                        </button>
                    </div>
                </div>

                <!-- Content -->
                <div class="px-6 py-6 text-center">
                    ${statusIcon}
                    
                    <h4 class="text-xl font-semibold mb-2 text-${statusColor}-600">
                        ${verificationResult.isValid ? 'Verification Successful' : 'Verification Failed'}
                    </h4>
                    
                    <p class="text-sm text-gray-600 mb-4">
                        <strong>File:</strong> ${fileName}
                    </p>
                    
                    <div class="bg-gray-50 rounded-lg p-4 mb-4 text-left">
                        <h5 class="font-medium text-gray-900 mb-2">Reason:</h5>
                        <p class="text-sm text-gray-700 mb-3">${verificationResult.reason}</p>
                        
                        <h5 class="font-medium text-gray-900 mb-2">Details:</h5>
                        <p class="text-sm text-gray-600 font-mono bg-white p-2 rounded border">
                            ${verificationResult.details}
                        </p>
                        
                        ${verificationResult.timestamp ? `
                            <div class="mt-3 pt-3 border-t border-gray-200">
                                <p class="text-xs text-gray-500">
                                    <strong>Generated:</strong> ${new Date(verificationResult.timestamp).toLocaleString()}
                                </p>
                            </div>
                        ` : ''}
                    </div>
                    
                    <div class="flex justify-center">
                        <button class="close-verification-modal text-white px-6 py-2 rounded-lg transition-colors duration-200" 
                                style="background-color: ${verificationResult.isValid ? '#10b981' : '#ef4444'};"
                                onmouseover="this.style.backgroundColor='${verificationResult.isValid ? '#059669' : '#dc2626'}'"
                                onmouseout="this.style.backgroundColor='${verificationResult.isValid ? '#10b981' : '#ef4444'}'">
                            Close
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    verificationModal.innerHTML = verificationHTML;
    
    // Add styles for verification modal
    const verificationStyle = document.createElement('style');
    verificationStyle.textContent = `
        .verification-modal {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1060;
        }
        
        .verification-modal-content {
            animation: verificationModalSlideIn 0.3s ease-out;
        }
        
        @keyframes verificationModalSlideIn {
            from {
                opacity: 0;
                transform: translateY(-20px) scale(0.95);
            }
            to {
                opacity: 1;
                transform: translateY(0) scale(1);
            }
        }
        
        .verification-modal .overlay {
            animation: verificationOverlayFadeIn 0.3s ease-out;
        }
        
        @keyframes verificationOverlayFadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
    `;
    
    shadowRoot.appendChild(verificationStyle);
    shadowRoot.appendChild(verificationModal);
    
    // Set up event listeners for closing the modal
    const closeBtns = verificationModal.querySelectorAll('.close-verification-modal');
    const overlay = verificationModal.querySelector('.overlay');
    
    const closeModal = () => {
        verificationModal.style.animation = 'verificationModalSlideOut 0.2s ease-in forwards';
        verificationModal.querySelector('.overlay').style.animation = 'verificationOverlayFadeOut 0.2s ease-in forwards';
        setTimeout(() => {
            verificationModal.remove();
            verificationStyle.remove();
        }, 200);
    };
    
    // Add event listeners to all close buttons (X button and Close button)
    closeBtns.forEach(btn => {
        btn.addEventListener('click', closeModal);
    });
    
    if (overlay) {
        overlay.addEventListener('click', closeModal);
    }
    
    // Add slide out animation styles
    const slideOutStyle = document.createElement('style');
    slideOutStyle.textContent = `
        @keyframes verificationModalSlideOut {
            from {
                opacity: 1;
                transform: translateY(0) scale(1);
            }
            to {
                opacity: 0;
                transform: translateY(-20px) scale(0.95);
            }
        }
        
        @keyframes verificationOverlayFadeOut {
            from { opacity: 1; }
            to { opacity: 0; }
        }
    `;
    shadowRoot.appendChild(slideOutStyle);
}

function createStatsModal(shadowRoot) {
    const modal = document.createElement('div');
    modal.className = 'stats-modal hidden fixed inset-0 z-50 overflow-y-auto';
    
    const baseHTML = `
        <div class="overlay fixed inset-0 bg-black bg-opacity-50"></div>
        <div class="min-h-screen px-4 text-center">
            <span class="inline-block h-screen align-middle" aria-hidden="true">&#8203;</span>
            
            <div class="modal-content inline-block w-full max-w-5xl my-8 text-left align-middle transition-all transform bg-white rounded-lg shadow-xl" 
                 style="max-height: fit-content;">
                <!-- Header -->
                <div class="sticky top-0 bg-white px-6 py-4 border-b border-gray-200 rounded-t-lg z-10">
                    <div class="flex justify-between items-center">
                        <h2 class="text-2xl font-bold">Quiz Performance Dashboard</h2>
                        <div class="flex items-center space-x-4">
                            <span id="badges-prompt" class="text-blue-300 font-semibold">Track your progress by taking quizzes! &nbsp;&nbsp;</span>
                            Badges:  &nbsp;
        <div id="badges" class="achievement-badges grid grid-cols-4 gap-2 items-center">
            <span class="text-sm font-semibold col-span-4">👾</span>
        </div>
                            <button class="close-modal text-gray-500 hover:text-gray-700">
                                <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                                </svg>
                            </button>
                        </div>
                    </div>
                </div>

                <!-- Content -->
                <div class="px-6 py-4 overflow-y-auto" style="max-height: calc(90vh - 180px);">
                    <div class="flex flex-col md:flex-row gap-6">
                        <!-- Left column: Chapter Progress -->
                        <div class="w-full md:w-2/5">
                            <div class="bg-gray-50 rounded-lg p-4">
                                <h3 class="text-lg font-semibold mb-4">Chapter Progress</h3>
                                
                                <!-- Info component for pages viewed -->
                                <div class="bg-blue-50 border border-blue-200 rounded-lg p-3 mb-4">
                                    <div class="flex items-start space-x-2">
                                        <svg class="w-5 h-5 text-blue-500 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                                        </svg>
                                        <div class="text-sm text-blue-700">
                                            <p class="font-medium mb-1">Pages Viewed So Far</p>
                                            <p class="text-blue-600 mb-2">This chart shows pages you've visited and their quiz progress. It will automatically update as you explore more content on the site.</p>
                                            <p class="text-blue-600"><strong>💡 Tip:</strong> Cumulative quizzes allow you to pass entire sections with a single quiz (score 60% or higher).</p>
                                        </div>
                                    </div>
                                </div>
                                <div style="height: 400px;"> <!-- Fixed height for chart -->
                                    <canvas id="chapterProgressChart"></canvas>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Right column: Existing charts -->
                        <div class="w-full md:w-3/5">
                            <div class="grid grid-cols-1 gap-6">
                                <div class="stats-card p-4 bg-gray-50 rounded-lg">
                                    <h3 class="text-lg font-semibold mb-4">Success Rate</h3>
                                    <div style="height: 250px; display: flex; align-items: center; justify-content: center;">
                                        <canvas id="successRateChart"></canvas>
                                    </div>
                                </div>
                                <div class="stats-card p-4 bg-gray-50 rounded-lg">
                                    <h3 class="text-lg font-semibold mb-4">Daily Quizzes</h3>
                                    <div style="height: 250px;">
                                        <canvas id="dailyQuizzesChart"></canvas>
                                    </div>
                                </div>
                                <div class="stats-card p-4 bg-gray-50 rounded-lg">
                                    <h3 class="text-lg font-semibold mb-4">Performance Trend</h3>
                                    <div style="height: 250px;">
                                        <canvas id="performanceTrendChart"></canvas>
                                    </div>
                                </div>
                                <div class="stats-card p-4 bg-gray-50 rounded-lg">
                                    <div class="grid grid-cols-2 gap-4">
                                        <div class="stat-box bg-white p-4 rounded-lg shadow">
                                            <h4 class="text-sm font-medium text-gray-500">Current Streak</h4>
                                            <p id="currentStreak" class="text-2xl font-bold"></p>
                                        </div>
                                        <div class="stat-box bg-white p-4 rounded-lg shadow">
                                            <h4 class="text-sm font-medium text-gray-500">Best Streak</h4>
                                            <p id="bestStreak" class="text-2xl font-bold"></p>
                                        </div>
                                        <div class="stat-box bg-white p-4 rounded-lg shadow">
                                            <h4 class="text-sm font-medium text-gray-500">Total Quizzes</h4>
                                            <p id="totalQuizzes" class="text-2xl font-bold"></p>
                                        </div>
                                        <div class="stat-box bg-white p-4 rounded-lg shadow">
                                            <h4 class="text-sm font-medium text-gray-500">Avg. Score</h4>
                                            <p id="avgScore" class="text-2xl font-bold"></p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Footer with action buttons - Now positioned at bottom -->
                <div class="sticky bottom-0 bg-white px-6 py-4 border-t border-gray-200 rounded-b-lg">
                    <div class="flex justify-end space-x-4">
                        <button id="download-pdf-btn" 
                                class="flex items-center space-x-2 bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg transition-colors duration-200">
                            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                                      d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                            </svg>
                            <span>Download Report</span>
                        </button>
                        
                        <div class="flex items-center">
                            <input type="file" 
                                   id="verify-pdf-input" 
                                   accept=".pdf" 
                                   class="hidden" />
                            <button id="verify-pdf-btn"
                                    class="flex items-center space-x-2 bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded-lg transition-colors duration-200">
                                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                                          d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                                </svg>
                                <span>Verify Report</span>
                            </button>
                            <div id="verification-result" class="ml-4 text-sm"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    modal.innerHTML = baseHTML;
    
    adjustChartContainer(modal);
    // Add styles
    const style = document.createElement('style');
    style.textContent = `
        ${badgeStyles}
        .stats-modal {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1050;
        }
        
        
        .stats-modal.hidden {
            opacity: 0;
            visibility: hidden;
        }
        .modal-content{
        top:10px;
        width: 1024px;
        }
        
        .stats-modal.hidden .modal-content {
            transform: translateY(20px);
        }
        
        .overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5);
        }
              /* Mobile-first responsive adjustments */
    @media (max-width: 768px) {
        .modal-content {
            margin: 0;
            width: 100%;
            height: 100%;
            max-height: 100vh;
            border-radius: 0;
        }
        
        .flex.flex-col.md\\:flex-row {
            flex-direction: column !important;
        }
        
        .w-full.md\\:w-2\\/5,
        .w-full.md\\:w-3\\/5 {
            width: 100% !important;
        }
        
        .stats-card {
            margin-bottom: 1rem;
        }
        
        .grid.grid-cols-2 {
            grid-template-columns: 1fr;
            gap: 0.5rem;
        }
    }
    #successRateChart {
        max-width: 250px; /* Or your preferred size */
        max-height: 250px;
        margin: auto;
    }
    
    /* Custom legend styles */
    .custom-legend {
        margin-top: 1rem;
    }
    
    .custom-legend .grid {
        display: grid;
        gap: 0.75rem;
    }
    
    .custom-legend .grid-cols-1 {
        grid-template-columns: repeat(1, minmax(0, 1fr));
    }
    
    .custom-legend .grid-cols-2 {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    
    @media (min-width: 640px) {
        .custom-legend .sm\\:grid-cols-2 {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }
    
    /* Ensure chart labels don't get clipped */
    .chartjs-render-monitor {
        overflow: visible !important;
    }
    `;
    
    shadowRoot.appendChild(style);
    shadowRoot.appendChild(modal);

    // Set up event listeners
    const setupModalListeners = () => {
        const closeBtn = modal.querySelector('.close-modal');
        const overlay = modal.querySelector('.overlay');
        
        if (closeBtn) {
            closeBtn.addEventListener('click', () => modal.classList.add('hidden'));
        }
        
        if (overlay) {
            overlay.addEventListener('click', () => modal.classList.add('hidden'));
        }
    };

    requestAnimationFrame(setupModalListeners);

    // Replace the download and verification button section with this:
        // Replace the action buttons section (around line 196)
        const actionButtons = `
        
    `;

    // Insert the action buttons
    const modalContent = modal.querySelector('.modal-content');
    modalContent.insertAdjacentHTML('beforeend', actionButtons);

    // Set up verification handling
    const verifyBtn = modal.querySelector('#verify-pdf-btn');
    const fileInput = modal.querySelector('#verify-pdf-input');
    const resultDiv = modal.querySelector('#verification-result');

    verifyBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        resultDiv.innerHTML = `
            <div class="flex items-center text-yellow-500">
                <svg class="animate-spin h-5 w-5 mr-2" viewBox="0 0 24 24">
                    <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none"></circle>
                    <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Verifying...
            </div>
        `;

        try {
            const verificationResult = await verifyPDFReport(file);
            showVerificationModal(shadowRoot, verificationResult, file.name);
            resultDiv.innerHTML = verificationResult.isValid ? 
                '<span class="text-green-500">✓ Valid Report</span>' :
                '<span class="text-red-500">✗ Invalid Report</span>';
        } catch (error) {
            console.error('Verification error:', error);
            resultDiv.innerHTML = '<span class="text-red-500">Error verifying report</span>';
            showVerificationModal(shadowRoot, {
                isValid: false,
                reason: 'An unexpected error occurred during verification.',
                details: error.message || 'Unknown error'
            }, file.name);
        }
    });

    // Modify download button to show hash after generation
    const downloadBtn = modal.querySelector('#download-pdf-btn');
    downloadBtn.addEventListener('click', async () => {
        try {
            const { hash } = await generateQuizPDF();
            resultDiv.innerHTML = `
                <span class="text-gray-600">
                    Report Hash: ${hash.substring(0, 8)}...
                </span>
            `;
        } catch (error) {
            console.error('Error generating PDF:', error);
            resultDiv.innerHTML = '<span class="text-red-500">Error generating report</span>';
        }
    });

    return modal;
}

function processStatsData(stats) {
    // Get last 7 days
    const dates = [...new Array(7)].map((_, i) => {
        const d = new Date();
        d.setDate(d.getDate() - i);
        return d.toISOString().split('T')[0];
    }).reverse();

    // Process daily stats
    const dailyStats = dates.map(date => {
        const dayStats = stats.allStats.filter(s => s.date === date);
        return {
            date,
            count: dayStats.length,
            correctRate: dayStats.length ? 
                dayStats.reduce((sum, s) => sum + (s.score.correct / s.score.total), 0) / dayStats.length : 0
        };
    });

    // Calculate streaks
    let currentStreak = 0;
    let bestStreak = 0;
    let tempStreak = 0;

    dailyStats.forEach((day, i) => {
        if (day.count > 0) {
            tempStreak++;
            if (i === dailyStats.length - 1) {
                currentStreak = tempStreak;
            }
            if (tempStreak > bestStreak) {
                bestStreak = tempStreak;
            }
        } else {
            if (i < dailyStats.length - 1) {
                tempStreak = 0;
            }
        }
    });

    return {
        dailyStats,
        currentStreak,
        bestStreak,
        totalQuizzes: stats.attempts,
        avgScore: ((stats.totalCorrect / stats.totalQuestions) * 100).toFixed(1)
    };
}
function createCharts(shadowRoot, processedStats) {
    updateBadges(shadowRoot); 
    const { dailyStats } = processedStats;

    try {
        // Success Rate Chart
        if (chartInstances.successRateChart) {
            chartInstances.successRateChart.destroy();
            chartInstances.successRateChart = null;
        }
        const successCtx = shadowRoot.querySelector('#successRateChart').getContext('2d');
        chartInstances.successRateChart = new Chart(successCtx, {
            type: 'doughnut',
            data: {
                labels: ['Correct', 'Incorrect'],
                datasets: [{
                    data: [processedStats.avgScore, 100 - processedStats.avgScore],
                    backgroundColor: ['#3b82f6', '#93c5fd'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                cutout: '70%',
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });

        // Daily Quizzes Chart
        if (chartInstances.dailyQuizzesChart) {
            chartInstances.dailyQuizzesChart.destroy();
            chartInstances.dailyQuizzesChart = null;
        }
        const dailyCtx = shadowRoot.querySelector('#dailyQuizzesChart').getContext('2d');
        chartInstances.dailyQuizzesChart = new Chart(dailyCtx, {
            type: 'bar',
            data: {
                labels: dailyStats.map(d => d.date.slice(5)),
                datasets: [{
                    label: 'Number of Quizzes',
                    data: dailyStats.map(d => d.count),
                    backgroundColor: '#3b82f6',
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            stepSize: 1
                        }
                    }
                }
            }
        });

        // Performance Trend Chart
        if (chartInstances.performanceTrendChart) {
            chartInstances.performanceTrendChart.destroy();
            chartInstances.performanceTrendChart = null;
        }
        const trendCtx = shadowRoot.querySelector('#performanceTrendChart').getContext('2d');
        chartInstances.performanceTrendChart = new Chart(trendCtx, {
            type: 'line',
            data: {
                labels: dailyStats.map(d => d.date.slice(5)),
                datasets: [{
                    label: 'Success Rate',
                    data: dailyStats.map(d => (d.correctRate * 100).toFixed(1)),
                    borderColor: '#3b82f6',
                    tension: 0.3,
                    fill: false
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            callback: value => `${value}%`
                        }
                    }
                }
            }
        });

        // Update stat boxes
        shadowRoot.querySelector('#currentStreak').textContent = `${processedStats.currentStreak} days`;
        shadowRoot.querySelector('#bestStreak').textContent = `${processedStats.bestStreak} days`;
        shadowRoot.querySelector('#totalQuizzes').textContent = processedStats.totalQuizzes;
        shadowRoot.querySelector('#avgScore').textContent = `${processedStats.avgScore}%`;

    } catch (error) {
        console.error('Error creating charts:', error);
    }
}
export function initStatsDisplay(shadowRoot) {
    const modal = createStatsModal(shadowRoot);
    const historyBtn = shadowRoot.querySelector('#chat-quiz-btn');
    enableTooltip(historyBtn, "View quiz stats and badges", shadowRoot);
    
    if (!historyBtn) {
        console.error('History button not found in shadow root');
        return;
    }

    historyBtn.addEventListener('click', async () => {
        modal.classList.remove('hidden');
        try {
            // Destroy existing charts before creating new ones
            destroyAllCharts();
            
            // Parse/update chapter data
            // parseNavigation();
            
            // Create all charts
            createProgressChart(shadowRoot);
            
            // Create other charts
            const stats = await getQuizStats();
            const processedStats = processStatsData(stats);
            createCharts(shadowRoot, processedStats);
        } catch (error) {
            console.error('Error loading stats:', error);
        }
    });

    // Add cleanup on modal close
    const closeBtn = modal.querySelector('.close-modal');
    const overlay = modal.querySelector('.overlay');

    const cleanup = () => {
        modal.classList.add('hidden');
        destroyAllCharts();
    };

    closeBtn.addEventListener('click', cleanup);
    overlay.addEventListener('click', cleanup);
}

export function getQuizStreak() {
    return parseInt(localStorage.getItem('quizStreak') || '0');
}

export function getTotalQuizzesTaken() {
    return parseInt(localStorage.getItem('totalQuizzes') || '0');
}

export function openStatsModal() {
    // Replace 'your-element-id' with the ID of the element hosting the shadow root
    const hostElement = document.getElementById('widget-chat-container');
    if (!hostElement || !hostElement.shadowRoot) {
        console.error('Host element or shadowRoot not found');
        return;
    }

    const modalButton = hostElement.shadowRoot.querySelector('#chat-quiz-btn');
    if (modalButton) {
        modalButton.click();
    } else {
        console.error('Modal button not found in shadow root', hostElement.shadowRoot);
    }
}


// Add a cleanup function
function destroyAllCharts() {
    Object.values(chartInstances).forEach(chart => {
        if (chart) {
            chart.destroy();
        }
    });
    chartInstances = {
        progressChart: null,
        successRateChart: null,
        dailyQuizzesChart: null,
        performanceTrendChart: null
    };
}


export function updateChapterQuizCount(chapterNumber) {
    try {
      const data = JSON.parse(localStorage.getItem(STORAGE_KEY_CHAPTERS));
      const chapter = data.find(c => c.chapter === chapterNumber);
      if (chapter && chapter.quizzesTaken < chapter.totalQuizzes) {
        chapter.quizzesTaken++;
        localStorage.setItem(STORAGE_KEY_CHAPTERS, JSON.stringify(data));
      }
    } catch (error) {
      console.error('Error updating quiz count:', error);
    }
  }

function createChartLoader(container) {
    const loader = document.createElement('div');
    loader.className = 'chart-loader';
    loader.innerHTML = `
        <div class="absolute inset-0 bg-white bg-opacity-75 flex items-center justify-center">
            <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
        </div>
    `;
    container.style.position = 'relative';
    return loader;
}

function createProgressChart(shadowRoot, chartId = 'chapterProgressChart') {
    try {
        const chartContainer = shadowRoot.querySelector(`#${chartId}`).parentElement;
        const loader = createChartLoader(chartContainer);
        chartContainer.appendChild(loader);

        if (chartInstances.progressChart) {
            chartInstances.progressChart.destroy();
            chartInstances.progressChart = null;
        }

        return getChapterProgress().then(async (chapterProgress) => {
            // Use new chapterMap-based system
            let data = [];
            try {
                // Try to get data from new chapterMap system
                data = await getDashboardChapterData();
                
                // If no chapterMap data, fallback to old localStorage system
                if (!data.length) {
                    console.warn('No chapterMap data available, falling back to localStorage');
                    const storedData = localStorage.getItem(STORAGE_KEY_CHAPTERS);
                    const parsedData = JSON.parse(storedData);
                    
                    // If data isn't an array, trigger parseNavigation to rebuild it
                    if (!Array.isArray(parsedData)) {
                        console.warn('Chapter data is not in array format, rebuilding...');
                        // Get the navigation element
                        const navElement = document.querySelector('#quarto-sidebar');
                        if (navElement) {
                            // Rebuild chapter data
                            data = await parseNavigation(navElement);
                        } else {
                            console.error('Navigation element not found for rebuilding chapter data');
                            data = parsedData && typeof parsedData === 'object' ? [parsedData] : [];
                        }
                    } else {
                        data = parsedData;
                    }
                }
            } catch (error) {
                console.error('Error handling chapter data:', error);
                data = [];
            }

            if (!data.length) {
                console.warn('No chapter data available');
                return;
            }

            const ctx = shadowRoot.querySelector(`#${chartId}`).getContext('2d');
            
            // Minimum visible percentage for the "completed" portion
            const MIN_VISIBLE_PERCENT = 2;
            
            // Convert to percentages with minimum visibility and quiz progress
            const chartData = data.map(chapter => {
                // Handle both old and new data structures
                let progress;
                if (chapterProgress.has(chapter.url)) {
                    // New chapterMap system - progress keyed by URL
                    progress = chapterProgress.get(chapter.url);
                } else if (chapterProgress.has(chapter.chapter)) {
                    // Old system - progress keyed by chapter number
                    progress = chapterProgress.get(chapter.chapter);
                } else {
                    // No progress found
                    progress = { passedQuizzes: 0 };
                }
                
                
                const actualPercentage = (progress.passedQuizzes / 10) * 100;
                
                return {
                    ...chapter,
                    completionPercentage: progress.completedByCumulativeQuiz ? 
                        100 : // Full completion for cumulative quiz
                        (progress.passedQuizzes > 0 ? 
                            actualPercentage : 
                            MIN_VISIBLE_PERCENT),
                    remainingPercentage: progress.completedByCumulativeQuiz ? 
                        0 : // No remaining for cumulative quiz completion
                        (progress.passedQuizzes > 0 ? 
                            100 - actualPercentage : 
                            100 - MIN_VISIBLE_PERCENT),
                    completedByCumulativeQuiz: progress.completedByCumulativeQuiz,
                    cumulativeQuizTitle: progress.cumulativeQuizTitle,
                    hasSectionQuizzes: (progress.passedQuizzes > 0 || progress.totalAttempted > 0) && !progress.completedByCumulativeQuiz
                };
            });

            const minHeightPerChapter = 50; // Increased for multi-line labels
            const totalHeight = Math.max(450, chartData.length * minHeightPerChapter + 100); // Extra space for legend
            ctx.canvas.parentNode.style.height = `${totalHeight}px`;

            chartInstances.progressChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: chartData.map(chapter => {
                        // Always use the title from chapterMap data
                        const title = chapter.title || `Chapter ${chapter.chapter}`;
                        // Split long titles into two lines
                        if (title.length > 30) {
                            const words = title.split(' ');
                            const midPoint = Math.ceil(words.length / 2);
                            const firstLine = words.slice(0, midPoint).join(' ');
                            const secondLine = words.slice(midPoint).join(' ');
                            return [firstLine, secondLine];
                        }
                        return title;
                    }),
                    datasets: [{
                        label: 'Completed',
                        data: chartData.map(chapter => chapter.completionPercentage),
                        backgroundColor: chartData.map(chapter => {
                            if (chapter.completedByCumulativeQuiz) {
                                return '#10b981'; // Green for cumulative completion
                            } else if (chapter.hasSectionQuizzes) {
                                return '#3b82f6'; // Blue for section quiz progress
                            } else {
                                return '#e5e7eb'; // Gray for no quizzes
                            }
                        }),
                        borderColor: chartData.map(chapter => {
                            if (chapter.completedByCumulativeQuiz) {
                                return '#059669'; // Dark green border
                            } else if (chapter.hasSectionQuizzes) {
                                return '#2563eb'; // Dark blue border
                            } else {
                                return '#9ca3af'; // Gray border
                            }
                        }),
                        borderWidth: 1,
                        order: 1
                    },
                    {
                        label: 'Remaining',
                        data: chartData.map(chapter => chapter.remainingPercentage),
                        backgroundColor: '#93c5fd',
                        borderColor: '#60a5fa',
                        borderWidth: 1,
                        order: 2
                    }]
                },
                options: {
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: false,
                    layout: {
                        padding: {
                            left: 40, // Increased left padding for labels
                            right: 20,
                            top: 10,
                            bottom: 10
                        }
                    },
                    plugins: {
                        legend: {
                            display: false // Hide default legend, we'll create custom one
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    const chapter = data[context.dataIndex];
                                    
                                    // Handle both old and new data structures
                                    let progress;
                                    if (chapterProgress.has(chapter.url)) {
                                        // New chapterMap system - progress keyed by URL
                                        progress = chapterProgress.get(chapter.url);
                                    } else if (chapterProgress.has(chapter.chapter)) {
                                        // Old system - progress keyed by chapter number
                                        progress = chapterProgress.get(chapter.chapter);
                                    } else {
                                        // No progress found
                                        progress = { passedQuizzes: 0 };
                                    }
                                    
                                    const actualPercentage = (progress.passedQuizzes / 10) * 100;
                                    
                                    if (context.dataset.label === 'Completed') {
                                        if (progress.completedByCumulativeQuiz) {
                                            return `✅ Completed by Cumulative Quiz: ${progress.cumulativeQuizTitle || 'Cumulative Quiz'}`;
                                        } else if (progress.passedQuizzes > 0) {
                                            return `📚 Section Quiz Progress: ${actualPercentage.toFixed(1)}% (${progress.passedQuizzes}/${progress.totalAttempted || 10} quizzes passed)`;
                                        } else {
                                            return '📖 No quizzes taken yet';
                                        }
                                    }
                                    
                                    if (progress.completedByCumulativeQuiz) {
                                        return '✅ Page completed via cumulative quiz';
                                    } else if (progress.passedQuizzes > 0) {
                                        return `📚 Section quizzes: ${progress.totalAttempted || 10} taken, ${progress.passedQuizzes} passed`;
                                    } else {
                                        return '📖 No quizzes taken on this page';
                                    }
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            stacked: true,
                            title: {
                                display: true,
                                text: 'Completion Percentage'
                            },
                            ticks: {
                                callback: function(value) {
                                    return value + '%';
                                }
                            },
                            max: 100
                        },
                        y: {
                            stacked: true,
                            beginAtZero: true,
                            ticks: {
                                padding: 15, // Increased padding for multi-line labels
                                font: {
                                    size: 10, // Slightly smaller font for better fit
                                    lineHeight: 1.2
                                },
                                maxRotation: 0, // Keep labels horizontal
                                callback: function(value, index) {
                                    const label = this.chart.data.labels[index];
                                    // Handle multi-line labels
                                    if (Array.isArray(label)) {
                                        return label;
                                    }
                                    return label;
                                }
                            }
                        }
                    },
                    barThickness: 20,
                    barPercentage: 0.8,
                    categoryPercentage: 0.9,
                    borderRadius: 4
                }
            });
            
            // Create custom legend
            createCustomLegend(chartContainer, chartData);
            
        }).finally(() => {
            // Remove loader when done
            const loader = chartContainer.querySelector('.chart-loader');
            if (loader) loader.remove();
        });

    } catch (error) {
        console.error('Error creating progress chart:', error);
    }
}

// Create custom legend with grid layout
function createCustomLegend(chartContainer, chartData) {
    // Remove existing custom legend if it exists
    const existingLegend = chartContainer.querySelector('.custom-legend');
    if (existingLegend) {
        existingLegend.remove();
    }
    
    // Analyze chart data to determine which legend items to show
    const hasCumulativeCompleted = chartData.some(chapter => chapter.completedByCumulativeQuiz);
    const hasSectionQuizzes = chartData.some(chapter => chapter.hasSectionQuizzes);
    const hasAnyProgress = chartData.some(chapter => chapter.completionPercentage > 2); // More than minimum visible percent
    
    const legendItems = [];
    
    if (hasCumulativeCompleted) {
        legendItems.push({
            color: '#10b981',
            label: 'Completed (Cumulative Quiz)',
            description: 'Pages completed via cumulative quiz'
        });
    }
    
    if (hasSectionQuizzes) {
        legendItems.push({
            color: '#3b82f6',
            label: 'Section Quiz Progress',
            description: 'Pages with section quiz progress'
        });
    }
    
    // Always show remaining, but also show a general "Progress" item if no specific types detected
    if (!hasCumulativeCompleted && !hasSectionQuizzes) {
        if (hasAnyProgress) {
            legendItems.push({
                color: '#3b82f6',
                label: 'Quiz Progress',
                description: 'Pages with quiz activity'
            });
        } else {
            // No quiz activity yet - show informational item
            legendItems.push({
                color: '#3b82f6',
                label: 'Quiz Progress',
                description: 'Start taking quizzes to see your progress here'
            });
        }
    }
    
    legendItems.push({
        color: '#93c5fd',
        label: 'Remaining',
        description: hasAnyProgress ? 'Remaining quiz progress' : 'Available quiz content'
    });
    
    // Create legend HTML
    const legendHTML = `
        <div class="custom-legend mt-4">
            <h4 class="text-sm font-medium text-gray-700 mb-3">Progress Legend</h4>
            <div class="grid grid-cols-1 sm:grid-cols-2 gap-3">
                ${legendItems.map(item => `
                    <div class="flex items-center space-x-3 p-2 bg-gray-50 rounded-lg">
                        <div class="w-4 h-4 rounded border-2" style="background-color: ${item.color}; border-color: ${item.color};"></div>
                        <div class="flex-1 min-w-0">
                            <div class="text-sm font-medium text-gray-900">${item.label}</div>
                            <div class="text-xs text-gray-500">${item.description}</div>
                        </div>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
    
    // Insert legend after the chart
    const chartElement = chartContainer.querySelector('#chapterProgressChart');
    chartElement.insertAdjacentHTML('afterend', legendHTML);
}

// Update the modal's chart container to accommodate the larger chart
function adjustChartContainer(modal) {
    const chartContainer = modal.querySelector('#chapterProgressChart').parentNode;
    chartContainer.style.minHeight = '600px'; // Increased minimum height
    chartContainer.style.overflow = 'auto';
    chartContainer.style.paddingRight = '10px'; // Add some padding for scrollbar
}
  
  
  // Modified modal HTML addition for the progress chart
  export function addChartToModal(modalHTML) {
    // Add this div before the grid of charts in your modal
    const chartSection = `
      <div class="w-full mb-6">
        <div class="bg-gray-50 rounded-lg p-4">
          <h3 class="text-lg font-semibold mb-4">Chapter Progress</h3>
          <div class="h-[400px]"> <!-- Fixed height container -->
            <canvas id="chapterProgressChart"></canvas>
          </div>
        </div>
      </div>
    `;
    
    // Insert the chart section before the grid
    return modalHTML.replace(
      '<div class="grid grid-cols-1 md:grid-cols-2 gap-6">',
      `${chartSection}<div class="grid grid-cols-1 md:grid-cols-2 gap-6">`
    );
  }


//   ADDING BADGES

function updateBadges(shadowRoot) {
    const badgesContainer = shadowRoot.querySelector('#badges');
    const badgesPrompt = shadowRoot.querySelector('#badges-prompt');
    
    // Hide prompt if any badges are earned
    if (getQuizStreak() > 0 || getTotalQuizzesTaken() > 0) {
        badgesPrompt.style.display = 'none';
    }
    
    const streak = getQuizStreak();
    const totalQuizzes = getTotalQuizzesTaken();
    
    // Get previously earned badges
    const lastBadgeState = JSON.parse(localStorage.getItem('lastBadgeState') || '{}');
    const currentBadgeState = {
        streak,
        totalQuizzes,
        firstQuiz: totalQuizzes > 0
    };
    
    // Check for new badges and show detailed popovers
    if (totalQuizzes > 0 && !lastBadgeState.firstQuiz) {
        showPopover(
            shadowRoot, 
            '🎯 First Quiz Badge Earned!\nAwarded for completing your first quiz. Keep going!', 
            'success',
            4000
        );
    }
    
    // Check for new streak badge with more detailed message
    if (streak > (lastBadgeState.streak || 0)) {
        showPopover(
            shadowRoot, 
            `🔥 New Streak Badge: ${streak} days!\nYou've maintained a perfect quiz score streak for ${streak} consecutive days!`, 
            'success',
            4000
        );
    }
    
    // Check for new milestone badges with more detailed message
    const lastMilestone = Math.floor((lastBadgeState.totalQuizzes || 0) / 10);
    const currentMilestone = Math.floor(totalQuizzes / 10);
    if (currentMilestone > lastMilestone) {
        showPopover(
            shadowRoot, 
            `🏆 Quiz Master Achievement!\nCongratulations on completing ${currentMilestone * 10} quizzes! Your dedication to learning is impressive!`, 
            'success',
            4000
        );
    }
    
    // Update badge state in localStorage
    localStorage.setItem('lastBadgeState', JSON.stringify(currentBadgeState));
    
    // Clear existing badges and add current ones
    badgesContainer.innerHTML = '';
    
    // First Quiz Badge
    if (totalQuizzes > 0) {
        addBadge(badgesContainer, '🎯', 'First Quiz Completed: Your journey begins!');
    }
    
    // Streak Badge
    if (streak > 0) {
        const streakBadge = document.createElement('div');
        streakBadge.className = 'streak-badge flex items-center justify-center w-6 h-6 rounded-full bg-blue-500 text-white text-xs font-bold cursor-help';
        streakBadge.textContent = streak;
        const streakDescription = `${streak} Day Perfect Score Streak: Keep the momentum going!`;
        streakBadge.title = streakDescription;
        
        // Add click handler for streak badge
        streakBadge.addEventListener('click', () => {
            showPopover(
                shadowRoot, 
                `🔥 ${streakDescription}\nMaintain your streak by completing quizzes with perfect scores!`, 
                'info',
                4000
            );
        });
        
        badgesContainer.appendChild(streakBadge);
    }
    
    // Quiz Count Badges
    const quizMilestones = Math.floor(totalQuizzes / 10);
    for (let i = 0; i < quizMilestones; i++) {
        const milestone = (i + 1) * 10;
        addBadge(
            badgesContainer, 
            '🏆', 
            `${milestone} Quiz Master: Completed ${milestone} quizzes!`
        );
    }
}
function addBadge(container, emoji, tooltip) {
    const badge = document.createElement('div');
    badge.className = 'badge text-xl cursor-help';
    badge.textContent = emoji;
    badge.title = tooltip;
    
    // Add click handler to show popover with badge info
    badge.addEventListener('click', () => {
        let description;
        switch (emoji) {
            case '🎯':
                description = 'First Quiz Badge: Awarded for completing your first quiz!';
                break;
            case '🔥':
                description = `Streak Badge: ${tooltip}`;
                break;
            case '🏆':
                description = `Achievement Badge: ${tooltip}`;
                break;
            default:
                description = tooltip;
        }
        showPopover(container.getRootNode(), `${emoji} ${description}`, 'info', 4000);
    });
    
    container.appendChild(badge);
}

function isNewBadge(tooltip) {
    const earnedBadges = JSON.parse(localStorage.getItem('earnedBadges') || '[]');
    if (!earnedBadges.includes(tooltip)) {
        earnedBadges.push(tooltip);
        localStorage.setItem('earnedBadges', JSON.stringify(earnedBadges));
        return true;
    }
    return false;
}


const badgeStyles = `
    .badge {
        transition: transform 0.2s;
        cursor: pointer;
    }
    .badge:hover {
        transform: scale(1.2);
    }
    .new-badge-glow {
        animation: glow 2s ease-in-out infinite;
    }
    @keyframes glow {
        0%, 100% { filter: drop-shadow(0 0 2px gold); }
        50% { filter: drop-shadow(0 0 8px gold); }
    }
    .streak-badge {
        animation: pop-in 0.3s ease-out;
        cursor: pointer;
        transition: transform 0.2s;
    }
    .streak-badge:hover {
        transform: scale(1.2);
    }
    @keyframes pop-in {
        0% { transform: scale(0); }
        50% { transform: scale(1.2); }
        100% { transform: scale(1); }
    }
`;
