# performance_monitor.py
import asyncio
import aiohttp
import time
import statistics
from datetime import datetime

class PerformanceMonitor:
    def __init__(self, api_base_url: str = "http://127.0.0.1:8000"):
        self.api_base_url = api_base_url
        
    async def measure_latency(self, session: aiohttp.ClientSession, product: str, iterations: int = 10):
        """Measure API latency for a specific product"""
        latencies = []
        
        for _ in range(iterations):
            start = time.time()
            try:
                async with session.post(
                    f"{self.api_base_url}/classify",
                    json={"designation": product}
                ) as response:
                    await response.json()
                    latency = (time.time() - start) * 1000
                    latencies.append(latency)
            except Exception as e:
                print(f"Error: {e}")
                
        return {
            'mean': statistics.mean(latencies),
            'median': statistics.median(latencies),
            'min': min(latencies),
            'max': max(latencies),
            'std': statistics.stdev(latencies) if len(latencies) > 1 else 0
        }
    
    async def stress_test(self, concurrent_requests: int = 20, duration_seconds: int = 60):
        """Stress test the API"""
        print(f"Starting stress test: {concurrent_requests} concurrent requests for {duration_seconds}s")
        
        test_products = [
            "Fromage camembert Normandie 250g",
            "Vin rouge Bordeaux 75cl",
            "Huile olive extra vierge 50cl",
            "Champagne brut 75cl",
            "Saumon fumé 200g"
        ]
        
        start_time = time.time()
        request_count = 0
        errors = 0
        
        async def make_request(session):
            nonlocal request_count, errors
            while time.time() - start_time < duration_seconds:
                try:
                    product = test_products[request_count % len(test_products)]
                    async with session.post(
                        f"{self.api_base_url}/classify",
                        json={"designation": product}
                    ) as response:
                        await response.json()
                        request_count += 1
                except Exception:
                    errors += 1
                    
                await asyncio.sleep(0.1)  # Small delay
        
        async with aiohttp.ClientSession() as session:
            tasks = [make_request(session) for _ in range(concurrent_requests)]
            await asyncio.gather(*tasks)
        
        actual_duration = time.time() - start_time
        rps = request_count / actual_duration
        
        print(f"Stress test results:")
        print(f"   Total requests: {request_count}")
        print(f"   Errors: {errors}")
        print(f"   Requests per second: {rps:.2f}")
        print(f"   Error rate: {errors/(request_count+errors)*100:.2f}%")

async def main():
    monitor = PerformanceMonitor()
    
    # Test latency
    async with aiohttp.ClientSession() as session:
        stats = await monitor.measure_latency(session, "Fromage camembert 250g")
        print(f"Latency stats (ms): {stats}")
    
    # Stress test
    await monitor.stress_test(concurrent_requests=10, duration_seconds=30)

if __name__ == "__main__":
    asyncio.run(main())