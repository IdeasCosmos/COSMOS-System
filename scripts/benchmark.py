"""
성능 벤치마크 스크립트
"""

import asyncio
import time
import statistics
import aiohttp
import numpy as np
from typing import List, Dict

class BenchmarkRunner:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = []
    
    async def run_single_request(self, session: aiohttp.ClientSession, text: str) -> Dict:
        """단일 요청 실행"""
        start = time.perf_counter()
        
        payload = {
            "text": text,
            "enable_resonance": True
        }
        
        headers = {"X-API-Key": "dev-key"}
        
        async with session.post(
            f"{self.base_url}/analyze-resonance",
            json=payload,
            headers=headers
        ) as response:
            result = await response.json()
            latency = (time.perf_counter() - start) * 1000
            
            return {
                "latency_ms": latency,
                "status": response.status,
                "result": result
            }
    
    async def run_concurrent_requests(
        self,
        texts: List[str],
        concurrency: int = 10
    ) -> List[Dict]:
        """동시 요청 실행"""
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            for i in range(0, len(texts), concurrency):
                batch = texts[i:i+concurrency]
                batch_tasks = [
                    self.run_single_request(session, text)
                    for text in batch
                ]
                results = await asyncio.gather(*batch_tasks)
                tasks.extend(results)
                
                # 쿨다운
                await asyncio.sleep(0.1)
            
            return tasks
    
    async def benchmark(self):
        """전체 벤치마크 실행"""
        print("🚀 Starting COSMOS Benchmark")
        
        # 테스트 데이터
        test_texts = [
            "오늘 정말 기쁘네요!",
            "슬픈 소식을 들었어요...",
            "화가 나서 참을 수가 없어",
            "무서운 영화를 봤는데 정말 무서웠어요",
            "놀라운 발견을 했습니다",
            "평온한 하루를 보내고 있어요",
            "복잡한 감정이 교차하는 순간이에요",
            "한국 특유의 정이 느껴지네요",
            "참 잘했네요... (반어법)",
            "멘붕이 왔어요 현타가 와요"
        ] * 10  # 100개 요청
        
        # 워밍업
        print("🔥 Warming up...")
        await self.run_concurrent_requests(test_texts[:5], 1)
        
        # 부하 테스트
        for concurrency in [1, 5, 10, 20]:
            print(f"\n📊 Testing with concurrency: {concurrency}")
            
            start = time.time()
            results = await self.run_concurrent_requests(
                test_texts,
                concurrency
            )
            total_time = time.time() - start
            
            # 통계 계산
            latencies = [r["latency_ms"] for r in results]
            successful = sum(1 for r in results if r["status"] == 200)
            
            stats = {
                "concurrency": concurrency,
                "total_requests": len(results),
                "successful": successful,
                "total_time": total_time,
                "rps": len(results) / total_time,
                "latency_mean": statistics.mean(latencies),
                "latency_median": statistics.median(latencies),
                "latency_p95": np.percentile(latencies, 95),
                "latency_p99": np.percentile(latencies, 99)
            }
            
            self.print_stats(stats)
            self.results.append(stats)
        
        return self.results
    
    def print_stats(self, stats: Dict):
        """통계 출력"""
        print(f"  ✅ Success: {stats['successful']}/{stats['total_requests']}")
        print(f"  ⚡ RPS: {stats['rps']:.2f}")
        print(f"  ⏱️ Latency (ms):")
        print(f"    - Mean: {stats['latency_mean']:.2f}")
        print(f"    - Median: {stats['latency_median']:.2f}")
        print(f"    - P95: {stats['latency_p95']:.2f}")
        print(f"    - P99: {stats['latency_p99']:.2f}")

async def main():
    runner = BenchmarkRunner()
    results = await runner.benchmark()
    
    print("\n" + "="*50)
    print("📈 Benchmark Complete!")
    print("="*50)
    
    # 목표 확인
    targets = {
        "latency_p50": 50,  # 목표: <50ms
        "latency_p99": 100,  # 목표: <100ms
        "rps": 100  # 목표: >100 RPS
    }
    
    best_result = max(results, key=lambda x: x['rps'])
    
    print(f"\n🏆 Best Performance:")
    print(f"  - RPS: {best_result['rps']:.2f} (Target: >{targets['rps']})")
    print(f"  - P50 Latency: {best_result['latency_median']:.2f}ms (Target: <{targets['latency_p50']}ms)")
    print(f"  - P99 Latency: {best_result['latency_p99']:.2f}ms (Target: <{targets['latency_p99']}ms)")
    
    # 목표 달성 여부
    if (best_result['latency_median'] < targets['latency_p50'] and
        best_result['latency_p99'] < targets['latency_p99'] and
        best_result['rps'] > targets['rps']):
        print("\n✅ All performance targets met!")
    else:
        print("\n⚠️ Some targets not met. Consider optimization.")

if __name__ == "__main__":
    asyncio.run(main())