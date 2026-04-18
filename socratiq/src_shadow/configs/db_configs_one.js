export const DB_CONFIGS = {
  name: "socratiqDB",
  version: 3,
  stores: [
    {
      name: "chapterSummaries",
      keyPath: "chapterId",
      indexes: [
        {
          name: "urlIndex",
          keyPath: "url",
        },
        {
          name: "lastUpdated",
          keyPath: "lastUpdated",
        },
      ],
    },
    {
      name: "tinyMLDB_chats",
      keyPath: "id",
      indexes: [],
    },
    {
      name: "quizTitles",
      keyPath: "url",
      indexes: [],
    },

    {
      name: "progressReports",
      keyPath: "id",
      indexes: [
        {
          name: "date",
          keyPath: "date",
        },
      ],
    },

    {
      name: "quizHighScores_",
      keyPath: "id",
      indexes: [
        {
          name: "quizTitle",
          keyPath: "quizTitle",
        },
        {
          name: "percentageScore",
          keyPath: "percentageScore",
        },
      ],
    },
    {
      name: "quizHistory",
      keyPath: "id",
      indexes: [
        {
          name: "date",
          keyPath: "date",
        },
      ],
    },
    {
      name: "quizAttempts",
      keyPath: "date",
      indexes: [
        {
          name: "dateIndex",
          keyPath: "date",
          unique: true,
          multiEntry: false,
        },
      ],
    },
    {
      name: "quizHighScores",
      keyPath: "quizTitle",
      indexes: [
        {
          name: "scoreIndex",
          keyPath: "percentageScore",
          unique: false,
          multiEntry: false,
        },
      ],
    },
    {
      name: "quizStats",
      keyPath: "id",
      indexes: [
        {
          name: "dateIndex",
          keyPath: "date",
          unique: false,
          multiEntry: false,
        },
        {
          name: "scoreIndex",
          keyPath: "score",
          unique: false,
          multiEntry: false,
        },
      ],
    },
    {
      name: "tinyMLQuizResults",
      keyPath: "id",
      indexes: [],
    },
    {
      name: "reportHistory",
      keyPath: "date",
      indexes: [
        {
          name: "dateIndex",
          keyPath: "date",
          unique: true,
          multiEntry: false,
        },
      ],
    },
    {
      name: "ongoingIncorrectQuestions",
      keyPath: "id",
      indexes: [
        {
          name: "dateIndex",
          keyPath: "date",
          unique: false,
          multiEntry: false,
        },
      ],
    },
    {
      name: "review_history",
      keyPath: "date",
      indexes: [],
    },
    {
      name: "chapters",
      keyPath: "chapter",
      indexes: [],
    },
    {
      name: "cards",
      keyPath: "id",
      indexes: [{ name: "chapter", keyPath: "chapter" }],
    },
    {
      name: "card_tags",
      keyPath: "id", // We'll use id = `${card_id}_${tag}`
      indexes: [
        { name: "card_id", keyPath: "card_id" },
        { name: "tag", keyPath: "tag" },
      ],
    },
    {
      name: "current_chapter",
      keyPath: "chapter",
      indexes: [],
    },
    {
      name: "chapterMap",
      keyPath: "url",
      indexes: [
        {
          name: "lastUpdated",
          keyPath: "lastUpdated",
        },
        {
          name: "title",
          keyPath: "title",
        },
      ],
    },
    {
      name: "pageCompletions",
      keyPath: "id",
      indexes: [
        {
          name: "pageUrl",
          keyPath: "pageUrl",
          unique: true,
        },
        {
          name: "completedAt",
          keyPath: "completedAt",
        },
      ],
    },
    {
      name: "cumulativeQuizScores",
      keyPath: "normalizedUrl",
      indexes: [
        {
          name: "quizTitle",
          keyPath: "quizTitle",
          unique: false,
          multiEntry: false,
        },
        {
          name: "scoreIndex",
          keyPath: "score.percentage",
          unique: false,
          multiEntry: false,
        },
        {
          name: "completedAt",
          keyPath: "completedAt",
          unique: false,
          multiEntry: false,
        },
        {
          name: "passed",
          keyPath: "score.passed",
          unique: false,
          multiEntry: false,
        },
        {
          name: "quizType",
          keyPath: "quizType",
          unique: false,
          multiEntry: false,
        },
        {
          name: "isCumulative",
          keyPath: "isCumulative",
          unique: false,
          multiEntry: false,
        },
      ],
    },
    {
      name: "sectionQuizScores",
      keyPath: "id",
      indexes: [
        {
          name: "pageUrl",
          keyPath: "pageUrl",
          unique: false,
          multiEntry: false,
        },
        {
          name: "quizTitle",
          keyPath: "quizTitle",
          unique: false,
          multiEntry: false,
        },
        {
          name: "scoreIndex",
          keyPath: "score.percentage",
          unique: false,
          multiEntry: false,
        },
        {
          name: "completedAt",
          keyPath: "completedAt",
          unique: false,
          multiEntry: false,
        },
        {
          name: "passed",
          keyPath: "score.passed",
          unique: false,
          multiEntry: false,
        },
        {
          name: "quizType",
          keyPath: "quizType",
          unique: false,
          multiEntry: false,
        },
      ],
    },
  ],
};
