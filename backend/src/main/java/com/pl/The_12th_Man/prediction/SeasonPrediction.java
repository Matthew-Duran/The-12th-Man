package com.pl.The_12th_Man.prediction;

import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "season_predictions")
public class SeasonPrediction {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "team_name")
    private String teamName;

    @Column(name = "predicted_points")
    private Double predictedPoints;

    @Column(name = "predicted_rank")
    private Integer predictedRank;

    private String season;

    @Column(name = "created_at")
    private LocalDateTime createdAt;

    // Constructors
    public SeasonPrediction() {}

    // Getters and setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public String getTeamName() { return teamName; }
    public void setTeamName(String teamName) { this.teamName = teamName; }

    public Double getPredictedPoints() { return predictedPoints; }
    public void setPredictedPoints(Double predictedPoints) { this.predictedPoints = predictedPoints; }

    public Integer getPredictedRank() { return predictedRank; }
    public void setPredictedRank(Integer predictedRank) { this.predictedRank = predictedRank; }

    public String getSeason() { return season; }
    public void setSeason(String season) { this.season = season; }

    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
}