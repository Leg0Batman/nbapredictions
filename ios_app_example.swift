import SwiftUI

@main
struct NbaPredictor: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

struct ContentView: View {
    @State private var games: [GamePrediction] = []
    @State private var isLoading = false
    @State private var error: String?
    
    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("🏀 NBA Predictions")
                    .font(.title2)
                    .fontWeight(.bold)
                Spacer()
                Button(action: fetchPredictions) {
                    Image(systemName: "arrow.clockwise")
                        .font(.headline)
                }
            }
            .padding()
            .background(Color.blue)
            .foregroundColor(.white)
            
            if isLoading {
                VStack {
                    Spacer()
                    ProgressView()
                    Text("Loading...")
                        .foregroundColor(.gray)
                        .padding()
                    Spacer()
                }
            } else if let error = error {
                VStack(spacing: 12) {
                    Text("⚠️ Error")
                        .font(.headline)
                        .foregroundColor(.red)
                    Text(error)
                        .font(.body)
                    Button("Retry") {
                        fetchPredictions()
                    }
                    .foregroundColor(.blue)
                }
                .padding()
            } else if games.isEmpty {
                VStack(spacing: 12) {
                    Text("📭 No predictions")
                        .font(.headline)
                    Text("Check back later!")
                        .font(.caption)
                        .foregroundColor(.gray)
                }
                .padding()
            } else {
                ScrollView {
                    VStack(spacing: 12) {
                        ForEach(games, id: \.matchup) { game in
                            GameCard(game: game)
                        }
                    }
                    .padding()
                }
            }
        }
        .onAppear {
            fetchPredictions()
        }
    }
    
    func fetchPredictions() {
        isLoading = true
        error = nil
        
        let urlString = "http://192.168.1.108:8000/api/predictions/formatted"
        guard let url = URL(string: urlString) else {
            error = "Invalid URL"
            isLoading = false
            return
        }
        
        URLSession.shared.dataTask(with: url) { data, _, networkError in
            DispatchQueue.main.async {
                isLoading = false
                
                if let networkError = networkError {
                    error = "Network error: \(networkError.localizedDescription)"
                    return
                }
                
                guard let data = data else {
                    error = "No data"
                    return
                }
                
                do {
                    let response = try JSONDecoder().decode(APIResponse.self, from: data)
                    games = response.games
                } catch {
                    self.error = "Failed to parse data"
                }
            }
        }.resume()
    }
}

struct GameCard: View {
    let game: GamePrediction
    
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(game.matchup)
                .font(.headline)
                .lineLimit(2)
            
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("WINNER")
                        .font(.caption)
                        .foregroundColor(.gray)
                    Text(game.winner)
                        .font(.subheadline)
                        .fontWeight(.semibold)
                }
                Spacer()
                VStack(alignment: .trailing, spacing: 4) {
                    Text("CONFIDENCE")
                        .font(.caption)
                        .foregroundColor(.gray)
                    Text("\(String(format: "%.1f", game.confidence_pct))%")
                        .font(.headline)
                }
            }
            
            VStack(alignment: .leading, spacing: 8) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Home: \(String(format: "%.1f", game.home_prob))%")
                        .font(.caption)
                    ProgressView(value: game.home_prob / 100)
                        .tint(.blue)
                }
                
                VStack(alignment: .leading, spacing: 4) {
                    Text("Away: \(String(format: "%.1f", game.away_prob))%")
                        .font(.caption)
                    ProgressView(value: game.away_prob / 100)
                        .tint(.orange)
                }
            }
            
            Text(game.confidence)
                .font(.caption2)
                .padding(6)
                .background(Color.gray.opacity(0.2))
                .cornerRadius(4)
        }
        .padding()
        .background(Color.white)
        .cornerRadius(10)
    }
}

struct GamePrediction: Codable {
    let matchup: String
    let winner: String
    let confidence: String
    let home_prob: Double
    let away_prob: Double
    let confidence_pct: Double
}

struct APIResponse: Codable {
    let success: Bool
    let games: [GamePrediction]
    let count: Int
}

#Preview {
    ContentView()
}
