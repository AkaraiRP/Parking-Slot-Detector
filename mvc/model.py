class ParkingSlot:
    def __init__(self, id: int, label: str, status: str, is_occupied: bool=False, is_occupied_confidence: float=0) -> None:
        self.id = id
        self.label = label
        self.status = status
        self.is_occupied = is_occupied
        self._is_occupied_confidence = is_occupied_confidence

    def __repr__(self):
        return f"ParkingSlot(id={self.id}, label={self.label}, occupied={self.is_occupied} (confidence: {self._is_occupied_confidence}), status={self.status})"


class ParkingLot:
    DEFAULT_CONFIDENCE_THRESHOLD = .9

    def __init__(self, 
                 id: int, 
                 name: str,
                 slots: list[ParkingSlot]=[],
                 status: str="N/A"
                 ) -> None:
        self.id = id
        self.name = name
        self.status = status
        self.slots = slots

        self.update_occupancy_status()

    def add_slot(self, slot: ParkingSlot) -> None:
        self.slots.append(slot)

    def update_occupancy_status(self, confidence_threshold: float=DEFAULT_CONFIDENCE_THRESHOLD) -> None:
        if len(self.slots) == 0:
            self.occupancy_status = "N/A"
        else:
            count = self.get_empty_slot_count(confidence_threshold=confidence_threshold)
            if count == 0:
                self.occupancy_status = "FULL PARKING"
            elif count == 1:
                self.occupancy_status = "1 FREE SLOT"
            else:
                self.occupancy_status = f"{count} FREE SLOTS"

    def get_empty_slot_count(self, confidence_threshold: float=DEFAULT_CONFIDENCE_THRESHOLD) -> int:
        count = 0
        for slot in self.slots:
            if slot.is_occupied and slot._is_occupied_confidence >= confidence_threshold:
                count += 1
        return count

    def __repr__(self) -> str:
        return f"ParkingLot(id={self.id}, name={self.name}, number_of_slots={self._number_of_slots}, occupancy_status={self.occupancy_status}, status={self.status})"


# TODO: optimize Camera and CameraView for python instead of directly copying from ERD
class Camera:
    def __init__(self, id: int, name: str, location: str, status: str, pixel_width: int, pixel_height: int) -> None:
        self.id = id
        self.name = name
        self.location = location
        self.status = status
        self.pixel_width = pixel_width
        self.pixel_height = pixel_height

    def __repr__(self):
        return f"Camera(id={self.id}, name={self.name}, location={self.location}, status={self.status}, resolution={self.pixel_width}x{self.pixel_height})"


# TODO: discuss how to implement CameraView (maybe )
class CameraView:
    def __init__(self, id: int, view_corners: tuple[tuple[int]], view_rotation: int, view_priority: int) -> None:
        self.id = id
        self.view_corners = view_corners
        self.view_rotation = view_rotation
        self.view_priority = view_priority

    def __repr__(self):
        return f"CameraView(id={self.id}, x_pos={self.view_x_pos}, y_pos={self.view_y_pos}, width={self.view_width}, height={self.view_height}, rotation={self.view_rotation}, priority={self.view_priority})"