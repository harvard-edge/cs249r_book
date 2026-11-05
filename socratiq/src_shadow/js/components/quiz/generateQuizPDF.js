import { jsPDF } from 'jspdf';
import { autoTable } from 'jspdf-autotable';
import { getAllHighScores, getQuizHistory } from './quiz-storage.js';
import { generateContentHash, uint8ArrayToBase64 } from '../../libs/utils/hashUtils.js';
// import { saveGeneratedHashContent } from '../../libs/utils/hashTestUtils.js';
import { generateUniqueId } from '../../libs/utils/utils.js';


async function generateQuizPDF() {
    const doc = new jsPDF();
    const highScores = await getAllHighScores();
    
    // Generate unique report ID first
    const reportId = generateUniqueId();
    
    // A chapter number alone is ambiguous across books. Preserve the recorded
    // page scope, and keep old title-only scores visibly unassigned.
    const chapterQuizzes = highScores.reduce((acc, score) => {
        const chapterMatch = score.quizTitle.match(/^(\d+)\./);
        const chapter = chapterMatch ? chapterMatch[1] : 'Other';
        let sourceUrl = '';
        try {
            const url = new URL(score.pageUrl);
            if (['https:', 'http:'].includes(url.protocol)) {
                sourceUrl = `${url.origin}${url.pathname}`.replace(/\/$/, '');
            }
        } catch {}
        const key = JSON.stringify([sourceUrl, chapter]);
        if (!acc.has(key)) acc.set(key, {chapter, sourceUrl, quizzes: []});
        acc.get(key).quizzes.push(score);
        return acc;
    }, new Map());

    function writeGroupHeading(group, y) {
        if (y > 235) { doc.addPage(); y = 20; }
        doc.setFontSize(16);
        doc.text(`Chapter ${group.chapter}`, 20, y);
        doc.setFontSize(9);
        const sourceLines = doc.splitTextToSize(`Source: ${group.sourceUrl || 'Unassigned source'}`, 170);
        doc.text(sourceLines, 20, y + 7);
        return y + 11 + sourceLines.length * 5;
    }

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
    // ASCII metadata survives PDF string escaping without altering stored IDs.
    const verificationPayload = `base64:${uint8ArrayToBase64(new TextEncoder().encode(contentToHash))}`;

    // Add hidden verification content with clear delimiters
    doc.setFontSize(1);  // Very small font
    doc.setTextColor(255, 255, 255);  // White text (invisible)
    doc.text('---VERIFY_CONTENT_START---\n', 10, 15, { visible: false });
    doc.text(`${verificationPayload}\n`, 10, 16, { visible: false });
    doc.text('---VERIFY_CONTENT_END---', 10, 17, { visible: false });
    doc.setTextColor(0, 0, 0);  // Reset to black
    doc.setFontSize(12);  // Reset font size

    // Generate hash before adding verification page
    const timestamp = new Date().toISOString();
    const hash = generateContentHash(contentToHash, timestamp);
    
    // Save for comparison
    // saveGeneratedHashContent(contentToHash, timestamp, hash);

    // For each chapter
    for (const group of chapterQuizzes.values()) {
        const {quizzes} = group;
        yPosition = writeGroupHeading(group, yPosition);

        for (const quiz of quizzes) {
            if (yPosition > 250) {
                doc.addPage();
                yPosition = 20;
            }

            const history = await getQuizHistory(quiz);
            
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

                autoTable(doc, {
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
    for (const group of chapterQuizzes.values()) {
        const {quizzes} = group;
        yPosition = writeGroupHeading(group, yPosition);

        for (const quiz of quizzes) {
            const history = await getQuizHistory(quiz);
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
                    answer.wasCorrect ? 'Correct' : 'Incorrect'
                ];
            });

            // Add questions table
            autoTable(doc, {
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
                        const isCorrect = cell.raw === 'Correct';
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

    doc.addPage();
    doc.setFontSize(16);
    doc.text('Report Metadata Checksum', 20, 20);

    doc.setFontSize(12);
    doc.text('This checksum checks the embedded report metadata for consistency.', 20, 40);
    doc.text('Metadata Checksum:', 20, 60);
    doc.setFont(undefined, 'bold');
    doc.text(hash, 20, 70);
    doc.setFont(undefined, 'normal');
    doc.text('Generated:', 20, 90);
    doc.text(timestamp, 20, 100);

    doc.text('It does not certify authorship, mastery, or the visible PDF contents.', 20, 120);

    doc.save('quiz-performance-report.pdf');


    return { hash, timestamp };
}

export { generateQuizPDF };
