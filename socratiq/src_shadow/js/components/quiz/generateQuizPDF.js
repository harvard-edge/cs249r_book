import { jsPDF } from 'jspdf';
import 'jspdf-autotable';  // For tables in PDF
import { getAllHighScores, getQuizHistory } from './quiz-storage.js';
import { generateContentHash } from '../../libs/utils/hashUtils.js';
// import { saveGeneratedHashContent } from '../../libs/utils/hashTestUtils.js';
import { generateUniqueId } from '../../libs/utils/utils.js';


async function generateQuizPDF() {
    const doc = new jsPDF();
    const highScores = await getAllHighScores();
    
    // Generate unique report ID first
    const reportId = generateUniqueId();
    
    // Group quizzes by chapter
    const chapterQuizzes = highScores.reduce((acc, score) => {
        const chapterMatch = score.quizTitle.match(/^(\d+)\./);
        const chapter = chapterMatch ? chapterMatch[1] : 'Other';
        if (!acc[chapter]) acc[chapter] = [];
        acc[chapter].push(score);
        return acc;
    }, {});

    // Title
    doc.setFontSize(20);
    doc.text('Quiz Performance Report', 20, 20);
    doc.setFontSize(12);
    doc.text(`Generated on: ${new Date().toLocaleDateString()}`, 20, 30);

    let yPosition = 40;

    // After getting highScores, normalize the data
    const normalizedHighScores = highScores.map(score => ({
        ...score,
        quizTitle: score.quizTitle.replace(/\s+/g, '')  // Remove all spaces
    }));

    const verificationData = {
        reportId,
        highScores: normalizedHighScores,
        generatedDate: new Date().toLocaleDateString()
    };
    const contentToHash = JSON.stringify(verificationData);

    // Add hidden verification content with clear delimiters
    doc.setFontSize(1);  // Very small font
    doc.setTextColor(255, 255, 255);  // White text (invisible)
    doc.text('---VERIFY_CONTENT_START---\n', 10, 15, { visible: false });
    doc.text(`${contentToHash}\n`, 10, 16, { visible: false });  // Add newline
    doc.text('---VERIFY_CONTENT_END---', 10, 17, { visible: false });
    doc.setTextColor(0, 0, 0);  // Reset to black
    doc.setFontSize(12);  // Reset font size

    // Generate hash before adding verification page
    const timestamp = new Date().toISOString();
    const hash = generateContentHash(contentToHash, timestamp);
    
    // Save for comparison
    // saveGeneratedHashContent(contentToHash, timestamp, hash);

    // For each chapter
    for (const [chapter, quizzes] of Object.entries(chapterQuizzes)) {
        if (yPosition > 250) {
            doc.addPage();
            yPosition = 20;
        }

        doc.setFontSize(16);
        doc.text(`Chapter ${chapter}`, 20, yPosition);
        yPosition += 10;

        for (const quiz of quizzes) {
            if (yPosition > 250) {
                doc.addPage();
                yPosition = 20;
            }

            const history = await getQuizHistory(quiz.quizTitle);
            
            doc.setFontSize(14);
            doc.text(quiz.quizTitle, 25, yPosition);
            yPosition += 10;

            // Quiz statistics
            doc.setFontSize(12);
            doc.text(`Best Score: ${quiz.percentageScore}%`, 30, yPosition);
            yPosition += 7;
            doc.text(`Total Attempts: ${history?.length || 0}`, 30, yPosition);
            yPosition += 10;

            // Create table for attempts
            if (history && history.length > 0) {
                const tableData = history.map(attempt => [
                    new Date(attempt.date).toLocaleDateString(),
                    `${attempt.score.correct}/${attempt.score.total}`,
                    `${((attempt.score.correct/attempt.score.total) * 100).toFixed(1)}%`
                ]);

                doc.autoTable({
                    startY: yPosition,
                    head: [['Date', 'Score', 'Percentage']],
                    body: tableData,
                    margin: { left: 30 },
                    tableWidth: 150
                });

                yPosition = doc.lastAutoTable.finalY + 10;
            }
        }
        yPosition += 10;
    }

    // New section - Detailed Questions and Answers
    doc.addPage();
    doc.setFontSize(18);
    doc.text('Detailed Questions and Answers by Chapter', 20, 20);
    yPosition = 30;

    // Group all questions by chapter
    for (const [chapter, quizzes] of Object.entries(chapterQuizzes)) {
        doc.setFontSize(16);
        doc.text(`Chapter ${chapter}`, 20, yPosition);
        yPosition += 10;

        for (const quiz of quizzes) {
            const history = await getQuizHistory(quiz.quizTitle);
            if (!history || history.length === 0) continue;

            // Get the most recent attempt for this quiz
            const latestAttempt = history[history.length - 1];
            
            doc.setFontSize(14);
            doc.text(`${quiz.quizTitle}`, 25, yPosition);
            yPosition += 10;

            // Create table data for questions
            const questionTableData = latestAttempt.answers.map(answer => {
                // Format answer choices as a list
                const choices = latestAttempt.quizData
                    .find(q => q.question === answer.question)
                    .answers.map(a => a.text)
                    .join('\n');

                return [
                    answer.question,
                    choices,
                    answer.correctAnswer,
                    answer.selectedAnswerText,
                    answer.wasCorrect ? '✓' : '✗'
                ];
            });

            // Add questions table
            doc.autoTable({
                startY: yPosition,
                head: [['Question', 'Answer Choices', 'Correct Answer', 'Your Answer', 'Result']],
                body: questionTableData,
                margin: { left: 20 },
                columnStyles: {
                    0: { cellWidth: 50 }, // Question column
                    1: { cellWidth: 40 }, // Choices column
                    2: { cellWidth: 35 }, // Correct answer column
                    3: { cellWidth: 35 }, // Selected answer column
                    4: { cellWidth: 15 }  // Result column
                },
                styles: {
                    overflow: 'linebreak',
                    cellPadding: 2,
                    fontSize: 8
                },
                headStyles: {
                    fillColor: [66, 135, 245],
                    fontSize: 9,
                    fontStyle: 'bold'
                },
                didDrawCell: (data) => {
                    // Add green/red background for correct/incorrect answers
                    if (data.section === 'body' && data.column.index === 4) {
                        const cell = data.cell;
                        const isCorrect = cell.raw === '✓';
                        doc.setFillColor(isCorrect ? 200 : 255, isCorrect ? 255 : 200, 200);
                        doc.rect(cell.x, cell.y, cell.width, cell.height, 'F');
                        doc.setTextColor(0);
                        doc.text(cell.raw, cell.x + cell.width / 2, cell.y + cell.height / 2, {
                            align: 'center',
                            baseline: 'middle'
                        });
                    }
                }
            });

            yPosition = doc.lastAutoTable.finalY + 10;

            // Add page if needed
            if (yPosition > 250) {
                doc.addPage();
                yPosition = 20;
            }
        }
    }

    // Add a specific marker before verification page
    doc.addPage();
    doc.setFontSize(8);
    doc.text('---CONTENT_END---', 10, 10, { visible: false }); // Hidden marker
    
    // Get content up to this point for hashing
    const pdfContent = doc.output('arraybuffer');
    const pdfBuffer = new Uint8Array(pdfContent);
    const contentStr = new TextDecoder().decode(pdfBuffer);
    
    const markerIndex = contentStr.indexOf('---CONTENT_END---');
    
    // Now add verification page
    doc.setFontSize(16);
    doc.text('--- BEGIN VERIFICATION PAGE ---', 20, 20);

    doc.setFontSize(12);
    doc.text('This page contains verification information for this report.', 20, 40);
    doc.text('Verification Code:', 20, 60);
    doc.setFont(undefined, 'bold');
    doc.text(hash, 20, 70);
    doc.setFont(undefined, 'normal');
    doc.text('Generated:', 20, 90);
    doc.text(timestamp, 20, 100);

    doc.text('--- END VERIFICATION PAGE ---', 20, 120);

    doc.save('quiz-performance-report.pdf');


    return { hash, timestamp };
}

export { generateQuizPDF };