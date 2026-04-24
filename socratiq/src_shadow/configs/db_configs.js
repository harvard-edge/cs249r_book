export const DB_CONFIGS = {
    chapterDB: {
      name: "tinyMLChapterDB",
      version: 1,
      stores: [
        {
          name: "chapterSummaries",
          keyPath: "chapterId",
          indexes: [
            {
              name: "urlIndex",
              keyPath: "url"
            },
            {
              name: "lastUpdated",
              keyPath: "lastUpdated"
            }
          ]
        }
      ]
    },
    progressDB: {
      name: "progressReports",
      version: 1,
      stores: [
        {
          name: "reports",
          keyPath: "id",
          indexes: [
            {
              name: "date",
              keyPath: "date"
            }
          ]
        }
      ]
    },
    quizDB: {
      name: "tinyMLQuizDB",
      version: 1,
      stores: [
        {
          name: "quizHighScores",
          keyPath: "id",
          indexes: [
            {
              name: "quizTitle",
              keyPath: "quizTitle"
            },
            {
              name: "percentageScore",
              keyPath: "percentageScore"
            }
          ]
        }
      ]
    }
  };